"""Mind Map — Neural network visualization for RL agents.

Captures activations, computes Grad-CAM, extracts decision data
from SB3 CnnPolicy in real-time during gameplay.
"""

import base64
import io
import logging
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from doom.environment import VizDoomGym
from doom.scenarios import get_scenario

logger = logging.getLogger(__name__)


class MindMapPlayer:
    """AI player with neural network visualization."""

    def __init__(self, socketio):
        self.socketio = socketio
        self.active_sessions = {}
        self._lock = threading.Lock()

    def start(self, session_id, model_path, scenario_key, speed=0.05):
        stop_flag = threading.Event()
        with self._lock:
            self.active_sessions[session_id] = {
                "stop_flag": stop_flag,
                "speed": speed,
            }
        thread = threading.Thread(
            target=self._worker,
            args=(session_id, model_path, scenario_key, speed, stop_flag),
            daemon=True,
        )
        thread.start()

    def stop(self, session_id):
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["stop_flag"].set()

    def set_speed(self, session_id, speed):
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["speed"] = max(0.01, min(0.5, speed))

    def _worker(self, session_id, model_path, scenario_key, speed, stop_flag):
        env = None
        try:
            model = PPO.load(model_path, device="cpu")
            policy = model.policy
            policy.eval()

            # ─── Register activation hooks ───
            activations = {}
            hooks = []

            cnn = policy.features_extractor.cnn
            conv_layers = []
            for i, layer in enumerate(cnn):
                if hasattr(layer, 'weight') and len(layer.weight.shape) == 4:
                    conv_layers.append((i, layer))

            for idx, (i, layer) in enumerate(conv_layers):
                name = f"conv{idx+1}"
                def make_hook(n):
                    def hook(module, inp, out):
                        activations[n] = out.detach()
                    return hook
                h = layer.register_forward_hook(make_hook(name))
                hooks.append(h)

            scenario_cfg = get_scenario(scenario_key)
            button_names = scenario_cfg.get("buttons", [])
            env = VizDoomGym(scenario_key, render=False)

            self.socketio.emit("mindmap_status", {
                "session_id": session_id, "status": "running",
                "scenario": scenario_key,
                "buttons": button_names,
                "conv_layers": [f"conv{i+1}" for i in range(len(conv_layers))],
            })

            frame_buffer = []  # Ring buffer for attention replay
            prev_action_probs = None
            attention_trail = []  # Grad-CAM centroid trail
            episode = 0

            while not stop_flag.is_set():
                episode += 1
                obs, info = env.reset()
                total_reward = 0
                step = 0
                done = False

                while not done and not stop_flag.is_set():
                    step += 1
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                    # ─── Forward pass (captures activations via hooks) ───
                    with torch.no_grad():
                        features = policy.features_extractor(obs_tensor)
                        latent_pi, latent_vf = policy.mlp_extractor(features)
                        action_logits = policy.action_net(latent_pi)
                        value = policy.value_net(latent_vf).item()

                    action_probs = F.softmax(action_logits, dim=-1).squeeze().numpy().tolist()
                    entropy = float(-(torch.tensor(action_probs) * torch.log(torch.tensor(action_probs) + 1e-8)).sum())
                    action = int(np.argmax(action_probs))

                    # ─── Grad-CAM on last conv layer ───
                    grad_cam_b64 = None
                    cached_cam_np = None
                    try:
                        grad_cam_b64, cached_cam_np = self._compute_grad_cam(
                            policy, obs_tensor, action, conv_layers[-1][1],
                            activations.get(f"conv{len(conv_layers)}")
                        )
                    except Exception:
                        pass

                    # ─── Serialize activation heatmaps ───
                    act_data = {}
                    for name, act in activations.items():
                        # Channel mean → spatial heatmap
                        heatmap = act.squeeze(0).mean(dim=0).numpy()
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                        heatmap_resized = cv2.resize(heatmap_uint8, (80, 60))
                        _, buf = cv2.imencode(".png", heatmap_resized)
                        act_data[name] = base64.b64encode(buf).decode("utf-8")

                        # Also send individual top channel maps for conv3
                        if name == f"conv{len(conv_layers)}":
                            channels = act.squeeze(0)  # [C, H, W]
                            # Top 8 most active channels
                            channel_means = channels.mean(dim=(1, 2))
                            top_idx = channel_means.argsort(descending=True)[:8]
                            channel_maps = []
                            for ci in top_idx:
                                ch = channels[ci].numpy()
                                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                                ch_uint8 = (ch * 255).astype(np.uint8)
                                ch_resized = cv2.resize(ch_uint8, (40, 30))
                                _, cbuf = cv2.imencode(".png", ch_resized)
                                channel_maps.append({
                                    "idx": int(ci),
                                    "mean": float(channel_means[ci]),
                                    "data": base64.b64encode(cbuf).decode("utf-8"),
                                })
                            act_data["top_channels"] = channel_maps

                    # ─── Step environment ───
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    # ─── Encode game frame ───
                    raw_frame = env.get_raw_frame()
                    frame_b64 = ""
                    if raw_frame is not None:
                        frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                        _, fbuf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frame_b64 = base64.b64encode(fbuf).decode("utf-8")

                    # ─── Advantage estimation ───
                    advantages = []
                    for a_idx in range(len(action_probs)):
                        advantages.append(round(action_probs[a_idx] - (1.0 / len(action_probs)), 4))

                    # ─── Store in frame buffer for replay ───
                    frame_entry = {
                        "step": step,
                        "action": action,
                        "action_probs": [round(p, 4) for p in action_probs],
                        "value": round(value, 3),
                        "entropy": round(entropy, 4),
                        "reward": round(float(reward), 2),
                    }
                    frame_buffer.append(frame_entry)
                    if len(frame_buffer) > 200:
                        frame_buffer = frame_buffer[-200:]

                    # ─── Attention centroid from Grad-CAM ───
                    attn_cx, attn_cy = 0.5, 0.5
                    if grad_cam_b64 and cached_cam_np is not None:
                        try:
                            cam_h, cam_w = cached_cam_np.shape
                            ys, xs = np.mgrid[:cam_h, :cam_w]
                            total_mass = cached_cam_np.sum() + 1e-8
                            attn_cx = float((xs * cached_cam_np).sum() / total_mass) / cam_w
                            attn_cy = float((ys * cached_cam_np).sum() / total_mass) / cam_h
                        except Exception:
                            pass
                    attention_trail.append({"x": round(attn_cx, 3), "y": round(attn_cy, 3)})
                    if len(attention_trail) > 30:
                        attention_trail = attention_trail[-30:]

                    # ─── Critical moment detection (KL divergence) ───
                    kl_div = 0.0
                    is_critical = False
                    if prev_action_probs is not None:
                        p = np.array(action_probs) + 1e-8
                        q = np.array(prev_action_probs) + 1e-8
                        kl_div = float((p * np.log(p / q)).sum())
                        is_critical = kl_div > 0.5
                    prev_action_probs = list(action_probs)

                    # ─── Latent vector (512-dim features for t-SNE) ───
                    latent = features.squeeze().detach().numpy()
                    # Send compressed: top 32 dims by variance + 2D PCA projection
                    latent_list = [round(float(v), 3) for v in latent[:32]]

                    # ─── Per-layer channel means for timeline heatmap ───
                    channel_means = {}
                    for name, act in activations.items():
                        ch = act.squeeze(0)  # [C, H, W]
                        means = ch.mean(dim=(1, 2))
                        top8_idx = means.argsort(descending=True)[:8]
                        channel_means[name] = [round(float(means[i]), 3) for i in top8_idx]

                    # ─── Emit everything ───
                    action_name = button_names[action] if action < len(button_names) else f"ACT_{action}"

                    self.socketio.emit("mindmap_frame", {
                        "session_id": session_id,
                        "frame": frame_b64,
                        "step": step,
                        "episode": episode,
                        "reward": round(float(reward), 2),
                        "total_reward": round(float(total_reward), 2),
                        "done": done,
                        "action": action,
                        "action_name": action_name,
                        "buttons": button_names,
                        # Decision data
                        "action_probs": [round(p, 4) for p in action_probs],
                        "value": round(value, 3),
                        "entropy": round(entropy, 4),
                        "advantages": advantages,
                        # Activation heatmaps
                        "activations": act_data,
                        "grad_cam": grad_cam_b64,
                        # NEW: Deep visualization data
                        "attention_trail": attention_trail,
                        "attn_cx": round(attn_cx, 3),
                        "attn_cy": round(attn_cy, 3),
                        "kl_div": round(kl_div, 4),
                        "is_critical": is_critical,
                        "channel_means": channel_means,
                        # Game info
                        "info": {k: round(float(v), 1) if isinstance(v, (int, float)) else v
                                 for k, v in info.items()},
                    })

                    with self._lock:
                        current_speed = self.active_sessions.get(session_id, {}).get("speed", speed)
                    time.sleep(current_speed)

                # Episode end
                self.socketio.emit("mindmap_episode_end", {
                    "session_id": session_id,
                    "episode": episode,
                    "total_reward": round(float(total_reward), 2),
                    "steps": step,
                })
                time.sleep(1.0)

        except Exception as e:
            logger.error(f"[MindMap {session_id}] Error: {e}", exc_info=True)
            self.socketio.emit("mindmap_status", {
                "session_id": session_id, "status": "error", "error": str(e),
            })
        finally:
            # Clean up hooks
            for h in hooks:
                h.remove()
            if env:
                try:
                    env.close()
                except Exception:
                    pass
            with self._lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            self.socketio.emit("mindmap_status", {
                "session_id": session_id, "status": "stopped",
            })

    def _compute_grad_cam(self, policy, obs_tensor, target_action, target_layer, cached_act):
        """Compute Grad-CAM. Returns (base64_image, raw_numpy_cam) tuple."""
        if cached_act is None:
            return None, None

        obs_input = obs_tensor.clone().requires_grad_(True)
        grads = {}

        def save_grad(module, grad_in, grad_out):
            grads["target"] = grad_out[0].detach()

        handle = target_layer.register_full_backward_hook(save_grad)

        features = policy.features_extractor(obs_input)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
        logits[0, target_action].backward()
        handle.remove()

        if "target" not in grads:
            return None, None

        weights = grads["target"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * cached_act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(obs_tensor.shape[2], obs_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_uint8 = (cam * 255).astype(np.uint8)
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_resized = cv2.resize(cam_colored, (320, 240))
        _, buf = cv2.imencode(".png", cam_resized, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return base64.b64encode(buf).decode("utf-8"), cam
