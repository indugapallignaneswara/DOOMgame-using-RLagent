"""AI Demo player - streams game frames to browser via SocketIO."""

import base64
import logging
import threading
import time

import cv2
import numpy as np
from stable_baselines3 import PPO

from doom.environment import VizDoomGym
from doom.scenarios import get_scenario

logger = logging.getLogger(__name__)


class AIPlayer:
    """Manages AI demo playback sessions with frame streaming."""

    def __init__(self, socketio):
        self.socketio = socketio
        self.active_demos = {}
        self._lock = threading.Lock()

    def start_demo(self, session_id, model_path, scenario_key, speed=0.05):
        """Start an AI demo in a background thread."""
        stop_flag = threading.Event()

        with self._lock:
            self.active_demos[session_id] = {
                "stop_flag": stop_flag,
                "status": "running",
                "speed": speed,
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
                        self.socketio.emit("game_frame", {
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
                        })

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
