"""AI Demo player - streams game frames to browser via SocketIO.

Optionally streams perturbation-based saliency overlays (Greydanus 2018) on
the same channel, so students can see what the agent is attending to. The
overlay is computed every K env steps to keep the demo FPS up; the client
re-draws the last overlay onto every frame so it feels continuous.
"""

import base64
import io
import logging
import threading
import time

import cv2
import numpy as np
from matplotlib import colormaps as _colormaps
from PIL import Image
from stable_baselines3 import PPO

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


def _encode_saliency_png(saliency: np.ndarray, cmap_name: str = "magma") -> str:
    """Encode a saliency map as a base64 PNG with alpha = saliency strength.

    The map's values become pixel alpha so low-saliency regions are transparent
    and the underlying game frame shows through. The client just draws the PNG
    on top of the canvas with `globalAlpha = opacity_slider`.
    """
    # saliency is (H, W) float in [0, 1]
    cmap = _colormaps.get_cmap(cmap_name)
    rgba = (cmap(saliency) * 255).astype(np.uint8)  # (H, W, 4)
    # Override alpha: floor low-saliency to 0, then scale
    alpha = np.where(saliency < SALIENCY_ALPHA_FLOOR, 0.0,
                     (saliency - SALIENCY_ALPHA_FLOOR) / (1.0 - SALIENCY_ALPHA_FLOOR))
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    # Upscale 4x so the 100x160 saliency lines up with the 640x480 game frame
    # without aliasing on the client side. PIL handles this faster than cv2.
    img = Image.fromarray(rgba, mode="RGBA")
    img = img.resize((img.size[0] * 4, img.size[1] * 4), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class AIPlayer:
    """Manages AI demo playback sessions with frame streaming."""

    def __init__(self, socketio):
        self.socketio = socketio
        self.active_demos = {}
        self._lock = threading.Lock()

    def start_demo(self, session_id, model_path, scenario_key, speed=0.05,
                   saliency_mode: str = "off",
                   saliency_every: int = DEFAULT_SALIENCY_EVERY):
        """Start an AI demo in a background thread.

        saliency_mode: "off" | "policy" | "value"
        """
        stop_flag = threading.Event()

        with self._lock:
            self.active_demos[session_id] = {
                "stop_flag": stop_flag,
                "status": "running",
                "speed": speed,
                "saliency_mode": saliency_mode,
                "saliency_every": saliency_every,
            }

        thread = threading.Thread(
            target=self._demo_worker,
            args=(session_id, model_path, scenario_key, speed, stop_flag),
            daemon=True,
        )
        thread.start()

    def set_demo_speed(self, session_id, speed):
        """Update the playback speed for a running demo."""
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["speed"] = max(0.01, min(0.5, speed))

    def set_saliency_mode(self, session_id, mode: str):
        """Update saliency mode mid-demo. mode in {off, policy, value}."""
        if mode not in ("off", "policy", "value"):
            return
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["saliency_mode"] = mode

    def _demo_worker(self, session_id, model_path, scenario_key, speed, stop_flag):
        """Background worker that runs the AI and streams frames."""
        env = None
        try:
            logger.info(f"[Demo {session_id}] Loading model: {model_path}")
            model = PPO.load(model_path)

            # Load scenario config to get button names
            scenario_cfg = get_scenario(scenario_key)
            button_names = scenario_cfg.get("buttons", [])

            env = VizDoomGym(scenario_key, render=False)

            # Lazily build saliency engine — it's only ~30 ms to construct
            # but we skip if mode=off, and we share the instance across episodes.
            sal_engine = PerturbationSaliency(
                model, SaliencyConfig(stride=LIVE_SALIENCY_STRIDE, smooth_sigma=2.0)
            )

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

                    # Read the current saliency settings (may have changed
                    # mid-demo via set_saliency_mode).
                    with self._lock:
                        sal_mode = self.active_demos.get(
                            session_id, {}).get("saliency_mode", "off")
                        sal_every = self.active_demos.get(
                            session_id, {}).get("saliency_every", DEFAULT_SALIENCY_EVERY)

                    # Compute saliency on the *current* obs (before step) so the
                    # heatmap matches the frame we're about to render.
                    saliency_png = None
                    if sal_mode != "off" and step % sal_every == 0:
                        try:
                            pol_map, val_map = sal_engine.both(obs)
                            chosen = pol_map if sal_mode == "policy" else val_map
                            cmap = "magma" if sal_mode == "policy" else "viridis"
                            saliency_png = _encode_saliency_png(chosen, cmap_name=cmap)
                        except Exception as sal_e:
                            logger.warning(f"[Demo {session_id}] saliency failed: {sal_e}")

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step += 1

                    # Determine action name
                    action_name = button_names[action_int] if action_int < len(button_names) else f"ACTION_{action_int}"

                    # Get raw frame and encode
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
                        self.socketio.emit("game_frame", payload)

                    # Use the potentially updated speed
                    with self._lock:
                        current_speed = self.active_demos.get(session_id, {}).get("speed", speed)
                    time.sleep(current_speed)

                # Episode finished
                self.socketio.emit("demo_episode_end", {
                    "session_id": session_id,
                    "episode": episode,
                    "total_reward": round(float(total_reward), 2),
                    "steps": step,
                })
                logger.info(f"[Demo {session_id}] Episode {episode}: reward={total_reward:.1f}")

                time.sleep(1.0)  # Brief pause between episodes

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
        """Encode RGB numpy array as base64 JPEG."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buffer).decode("utf-8")

    def stop_demo(self, session_id):
        with self._lock:
            if session_id in self.active_demos:
                self.active_demos[session_id]["stop_flag"].set()
