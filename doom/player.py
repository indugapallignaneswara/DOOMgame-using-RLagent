"""AI Demo player - streams game frames to browser via SocketIO.

Optionally streams perturbation-based saliency overlays (Greydanus 2018) on
the same channel, so students can see what the agent is attending to. The
overlay is computed every K env steps to keep the demo FPS up; the client
re-draws the last overlay onto every frame so it feels continuous.

Belief Inspector
================
Each game_frame additionally carries the policy's full action distribution
(`action_probs`) and a scalar value estimate (`value`). For PPO/A2C these come
from the ActorCriticPolicy directly; for DQN we softmax the Q-vector
(temperature 1.0) for `action_probs` and emit max(Q) as `value`. Iyer's review
demands policy and value remain visually distinct downstream — we only emit
the raw numbers here, never blend.

Prediction-before-reveal mode (Kapoor) pauses the demo on every Nth step,
emits the top-3 action indices, and waits up to ~5s on a threading.Event
that the SocketIO `prediction_response` handler sets. After the user picks,
the demo resumes and the next game_frame is paired with a `prediction_result`
carrying correctness + running score.
"""

import base64
import io
import logging
import threading
import time

import cv2
import numpy as np
import torch
from matplotlib import colormaps as _colormaps
from PIL import Image
from stable_baselines3 import A2C, DQN, PPO

from doom.environment import VizDoomGym
from doom.scenarios import get_scenario
from doom.saliency import PerturbationSaliency, SaliencyConfig

logger = logging.getLogger(__name__)

# How often (in env steps) to recompute saliency. Tradeoff: lower = laggy demo,
# higher = stale overlay. 4 means the overlay updates ~5 times/sec at 20 FPS.
DEFAULT_SALIENCY_EVERY = 4

# Stride for the perturbation grid when computing saliency in live mode.
# The CLI demo uses stride=8 for quality; live mode uses 16 for ~4x speedup.
LIVE_SALIENCY_STRIDE = 16

# Saliency below this fraction of the per-frame max becomes fully transparent
# in the emitted PNG — keeps the overlay clean instead of painting everything.
SALIENCY_ALPHA_FLOOR = 0.25

# Belief Inspector defaults
DEFAULT_PREDICTION_EVERY = 8       # pause every Nth step in prediction mode
PREDICTION_FREEZE_MS = 800         # how long the no-overlay preview lingers
PREDICTION_RESPONSE_TIMEOUT = 30.0 # max wait for the user before auto-skip


def _encode_saliency_png(saliency: np.ndarray, cmap_name: str = "magma") -> str:
    """Encode a saliency map as a base64 PNG with alpha = saliency strength.

    The map's values become pixel alpha so low-saliency regions are transparent
    and the underlying game frame shows through. The client just draws the PNG
    on top of the canvas with `globalAlpha = opacity_slider`.
    """
    cmap = _colormaps.get_cmap(cmap_name)
    rgba = (cmap(saliency) * 255).astype(np.uint8)
    alpha = np.where(saliency < SALIENCY_ALPHA_FLOOR, 0.0,
                     (saliency - SALIENCY_ALPHA_FLOOR) / (1.0 - SALIENCY_ALPHA_FLOOR))
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    img = img.resize((img.size[0] * 4, img.size[1] * 4), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_model(model_path: str):
    """Try PPO, then A2C, then DQN. SB3 zips don't carry algo metadata reliably,
    so we fall through on the well-defined load errors instead of guessing."""
    last_err = None
    for cls in (PPO, A2C, DQN):
        try:
            return cls.load(model_path)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load model {model_path}: {last_err}")


@torch.no_grad()
def _compute_belief(model, obs: np.ndarray):
    """Return (action_probs: list[float], value: float) for one HWC obs.

    PPO/A2C use the ActorCritic head; DQN softmaxes its Q-vector and uses
    max(Q) as the scalar value. Caller wraps this in a try/except — a forward
    failure must never crash the demo loop.
    """
    device = next(model.policy.parameters()).device
    obs_t = torch.as_tensor(obs[None]).permute(0, 3, 1, 2).float().to(device) / 255.0

    if isinstance(model, DQN):
        q_vals = model.q_net(obs_t).squeeze(0)
        probs = torch.softmax(q_vals, dim=-1).cpu().tolist()
        value = float(q_vals.max().item())
        return probs, value

    features = model.policy.extract_features(obs_t)
    if isinstance(features, tuple):
        pi_feat, vf_feat = features
    else:
        pi_feat = vf_feat = features
    latent_pi = model.policy.mlp_extractor.forward_actor(pi_feat)
    latent_vf = model.policy.mlp_extractor.forward_critic(vf_feat)
    logits = model.policy.action_net(latent_pi)
    value = float(model.policy.value_net(latent_vf).squeeze(-1).item())
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return probs, value


class AIPlayer:
    """Manages AI demo playback sessions with frame streaming."""

    def __init__(self, socketio):
        self.socketio = socketio
        self.active_demos = {}
        self._lock = threading.Lock()

    def start_demo(self, session_id, model_path, scenario_key, speed=0.05,
                   saliency_mode: str = "off",
                   saliency_every: int = DEFAULT_SALIENCY_EVERY,
                   prediction_mode: bool = False,
                   prediction_every: int = DEFAULT_PREDICTION_EVERY):
        """Start an AI demo in a background thread.

        saliency_mode: "off" | "policy" | "value"
        prediction_mode: when True, pauses every Nth step and waits for the
            user to predict the action before revealing.
        """
        stop_flag = threading.Event()
        # The worker waits on this event when paused for a prediction; the
        # SocketIO handler `prediction_response` sets it once the user clicks.
        prediction_response_event = threading.Event()

        with self._lock:
            self.active_demos[session_id] = {
                "stop_flag": stop_flag,
                "status": "running",
                "speed": speed,
                "saliency_mode": saliency_mode,
                "saliency_every": saliency_every,
                "prediction_mode": prediction_mode,
                "prediction_every": prediction_every,
                "prediction_response_event": prediction_response_event,
                "prediction_response": None,
                "predictions_correct": 0,
                "predictions_total": 0,
            }

        thread = threading.Thread(
            target=self._demo_worker,
            args=(session_id, model_path, scenario_key, speed, stop_flag),
            daemon=True,
        )
        thread.start()

    def set_demo_speed(self, session_id, speed):
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["speed"] = max(0.01, min(0.5, speed))

    def set_saliency_mode(self, session_id, mode: str):
        if mode not in ("off", "policy", "value"):
            return
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["saliency_mode"] = mode

    def set_prediction_mode(self, session_id, enabled: bool, every: int | None = None):
        with self._lock:
            d = self.active_demos.get(session_id)
            if not d:
                return
            d["prediction_mode"] = bool(enabled)
            if every is not None:
                d["prediction_every"] = max(2, int(every))
            if not enabled:
                # Unblock a worker currently waiting on a prediction response.
                d["prediction_response"] = None
                d["prediction_response_event"].set()

    def submit_prediction(self, session_id, choice):
        """Called by the SocketIO `prediction_response` handler."""
        with self._lock:
            d = self.active_demos.get(session_id)
            if not d:
                return
            d["prediction_response"] = choice
            d["prediction_response_event"].set()

    def _demo_worker(self, session_id, model_path, scenario_key, speed, stop_flag):
        env = None
        try:
            logger.info(f"[Demo {session_id}] Loading model: {model_path}")
            model = _load_model(model_path)
            is_dqn = isinstance(model, DQN)
            logger.info(f"[Demo {session_id}] Loaded {type(model).__name__}")

            scenario_cfg = get_scenario(scenario_key)
            button_names = scenario_cfg.get("buttons", [])

            env = VizDoomGym(scenario_key, render=False)

            # Saliency engine; PerturbationSaliency uses the actor-critic head
            # directly so it's PPO/A2C only. Skip construction for DQN — the
            # Brain-cam dropdown will simply produce no overlay there.
            sal_engine = None
            if not is_dqn:
                try:
                    sal_engine = PerturbationSaliency(
                        model, SaliencyConfig(stride=LIVE_SALIENCY_STRIDE, smooth_sigma=2.0)
                    )
                except Exception as sal_e:
                    logger.warning(f"[Demo {session_id}] saliency init failed: {sal_e}")

            episode = 0
            while not stop_flag.is_set():
                episode += 1
                obs, info = env.reset()
                total_reward = 0
                step = 0
                done = False

                self.socketio.emit("demo_episode_start", {
                    "session_id": session_id,
                    "episode": episode,
                })

                while not done and not stop_flag.is_set():
                    action, _ = model.predict(obs, deterministic=True)
                    action_int = int(action)

                    with self._lock:
                        sal_mode = self.active_demos.get(
                            session_id, {}).get("saliency_mode", "off")
                        sal_every = self.active_demos.get(
                            session_id, {}).get("saliency_every", DEFAULT_SALIENCY_EVERY)
                        pred_mode = self.active_demos.get(
                            session_id, {}).get("prediction_mode", False)
                        pred_every = self.active_demos.get(
                            session_id, {}).get("prediction_every", DEFAULT_PREDICTION_EVERY)

                    # Belief: action distribution + V(s). Cached per-step on
                    # the demo dict so prediction mode and the frame emission
                    # share a single forward pass.
                    action_probs = None
                    value_est = None
                    try:
                        action_probs, value_est = _compute_belief(model, obs)
                    except Exception as be:
                        logger.warning(f"[Demo {session_id}] belief forward failed: {be}")

                    saliency_png = None
                    if sal_engine is not None and sal_mode != "off" and step % sal_every == 0:
                        try:
                            pol_map, val_map = sal_engine.both(obs)
                            chosen = pol_map if sal_mode == "policy" else val_map
                            cmap = "magma" if sal_mode == "policy" else "viridis"
                            saliency_png = _encode_saliency_png(chosen, cmap_name=cmap)
                        except Exception as sal_e:
                            logger.warning(f"[Demo {session_id}] saliency failed: {sal_e}")

                    # ─── Prediction-before-reveal pause ───────────────────
                    # Only pause when we have a usable belief vector — emitting
                    # a prediction request without action_probs would mean the
                    # client can't compute "correct vs not" in the reveal.
                    pending_prediction = False
                    if pred_mode and action_probs is not None and step > 0 \
                            and step % pred_every == 0:
                        try:
                            top3 = sorted(range(len(action_probs)),
                                          key=lambda i: action_probs[i], reverse=True)[:3]
                            with self._lock:
                                d = self.active_demos[session_id]
                                d["prediction_response"] = None
                                d["prediction_response_event"].clear()

                            # Send the current frame with overlays *suppressed*
                            # so the player sees only the raw game.
                            raw_frame = env.get_raw_frame()
                            if raw_frame is not None:
                                self.socketio.emit("prediction_request", {
                                    "session_id": session_id,
                                    "step": step,
                                    "frame": self._encode_frame(raw_frame),
                                    "top3_actions": top3,
                                    "buttons": button_names,
                                    "freeze_ms": PREDICTION_FREEZE_MS,
                                })
                            pending_prediction = True
                        except Exception as pe:
                            logger.warning(f"[Demo {session_id}] prediction setup failed: {pe}")
                            pending_prediction = False

                    if pending_prediction:
                        ev = self.active_demos[session_id]["prediction_response_event"]
                        ev.wait(timeout=PREDICTION_RESPONSE_TIMEOUT)
                        with self._lock:
                            d = self.active_demos[session_id]
                            user_choice = d.get("prediction_response")
                            # User may have toggled prediction mode off while we waited.
                            still_on = d.get("prediction_mode", False)
                        if still_on and user_choice is not None and user_choice != "skip":
                            try:
                                user_idx = int(user_choice)
                            except (TypeError, ValueError):
                                user_idx = -1
                            correct = (user_idx == action_int)
                            with self._lock:
                                d = self.active_demos[session_id]
                                d["predictions_total"] += 1
                                if correct:
                                    d["predictions_correct"] += 1
                                score_c = d["predictions_correct"]
                                score_t = d["predictions_total"]
                            self.socketio.emit("prediction_result", {
                                "session_id": session_id,
                                "step": step,
                                "user_choice": user_idx,
                                "actual_action": action_int,
                                "correct": correct,
                                "correct_count": score_c,
                                "total_count": score_t,
                            })

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step += 1

                    action_name = button_names[action_int] if action_int < len(button_names) else f"ACTION_{action_int}"

                    raw_frame = env.get_raw_frame()
                    if raw_frame is not None:
                        frame_jpg = self._encode_frame(raw_frame)
                        payload = {
                            "session_id": session_id,
                            "frame": frame_jpg,
                            "reward": round(float(reward), 2),
                            "total_reward": round(float(total_reward), 2),
                            "step": step,
                            "episode": episode,
                            "done": done,
                            "action": action_int,
                            "action_name": action_name,
                            "buttons": button_names,
                            "info": {k: round(float(v), 1) if isinstance(v, (int, float)) else v
                                     for k, v in info.items()},
                            "saliency_mode": sal_mode,
                        }
                        if saliency_png is not None:
                            payload["saliency_png"] = saliency_png
                        if action_probs is not None:
                            payload["action_probs"] = [round(p, 4) for p in action_probs]
                        if value_est is not None:
                            payload["value"] = round(value_est, 4)
                        self.socketio.emit("game_frame", payload)

                    with self._lock:
                        current_speed = self.active_demos.get(session_id, {}).get("speed", speed)
                    time.sleep(current_speed)

                self.socketio.emit("demo_episode_end", {
                    "session_id": session_id,
                    "episode": episode,
                    "total_reward": round(float(total_reward), 2),
                    "steps": step,
                })
                logger.info(f"[Demo {session_id}] Episode {episode}: reward={total_reward:.1f}")

                time.sleep(1.0)

        except Exception as e:
            logger.error(f"[Demo {session_id}] Error: {e}")
            self.socketio.emit("demo_status", {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
            })
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            with self._lock:
                if session_id in self.active_demos:
                    self.active_demos[session_id]["status"] = "stopped"

            self.socketio.emit("demo_status", {
                "session_id": session_id,
                "status": "stopped",
            })

    def _encode_frame(self, frame_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buffer).decode("utf-8")

    def stop_demo(self, session_id):
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["stop_flag"].set()
                # Also unblock any pending prediction wait.
                ev = self.active_demos[session_id].get("prediction_response_event")
                if ev is not None:
                    ev.set()
