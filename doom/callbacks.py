"""Training callbacks for SocketIO-based progress reporting."""

import os
import time
import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class WebTrainingCallback(BaseCallback):
    """SB3 callback that emits training metrics via SocketIO and saves checkpoints.

    Supports pause/resume via threading.Event and stop via threading.Event flag.
    """

    def __init__(self, socketio, session_id, total_timesteps=0, emit_freq=50,
                 check_freq=10000, save_path="train", pause_event=None,
                 stop_flag=None, verbose=0):
        super().__init__(verbose)
        self.socketio = socketio
        self.session_id = session_id
        self._total_timesteps_target = total_timesteps
        self.emit_freq = emit_freq
        self.check_freq = check_freq
        self.save_path = save_path
        self.pause_event = pause_event
        self.stop_flag = stop_flag
        self._start_time = None
        self._last_emit_time = 0
        self._episode_rewards = []
        self._recent_rewards = []
        self._episode_lengths = []
        self._recent_lengths = []

    def _on_training_start(self):
        self._start_time = time.time()
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Check stop flag
        if self.stop_flag is not None and self.stop_flag.is_set():
            logger.info(f"[{self.session_id}] Training stopped by user")
            return False

        # Check pause
        if self.pause_event is not None and not self.pause_event.is_set():
            logger.info(f"[{self.session_id}] Training paused...")
            self.pause_event.wait()
            logger.info(f"[{self.session_id}] Training resumed")

        # Collect episode rewards from locals
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self._episode_rewards.append(ep_reward)
                    self._recent_rewards.append(ep_reward)
                    self._episode_lengths.append(ep_length)
                    self._recent_lengths.append(ep_length)
                    if len(self._recent_rewards) > 100:
                        self._recent_rewards = self._recent_rewards[-100:]
                    if len(self._recent_lengths) > 100:
                        self._recent_lengths = self._recent_lengths[-100:]

        # Emit metrics periodically
        if self.n_calls % self.emit_freq == 0:
            elapsed = time.time() - self._start_time if self._start_time else 1
            fps = self.num_timesteps / max(elapsed, 0.001)
            total = self._total_timesteps_target or getattr(self.model, '_total_timesteps', 0)

            metrics = {
                "session_id": self.session_id,
                "timesteps": self.num_timesteps,
                "total_timesteps": total,
                "fps": round(fps, 1),
                "elapsed_time": round(elapsed, 1),
                "total_episodes": len(self._episode_rewards),
            }

            if self._recent_rewards:
                metrics["ep_reward_mean"] = round(
                    sum(self._recent_rewards) / len(self._recent_rewards), 2)
                metrics["ep_reward_latest"] = round(self._recent_rewards[-1], 2)

            if self._recent_lengths:
                metrics["ep_len_mean"] = round(
                    sum(self._recent_lengths) / len(self._recent_lengths), 0)

            # Extract losses and entropy from SB3 logger
            if hasattr(self, "logger") and self.logger is not None:
                try:
                    nv = self.logger.name_to_value
                    if "train/loss" in nv:
                        metrics["loss"] = round(nv["train/loss"], 4)
                    if "train/policy_gradient_loss" in nv:
                        metrics["policy_loss"] = round(nv["train/policy_gradient_loss"], 6)
                    if "train/value_loss" in nv:
                        metrics["value_loss"] = round(nv["train/value_loss"], 4)
                    if "train/entropy_loss" in nv:
                        metrics["entropy_loss"] = round(nv["train/entropy_loss"], 4)
                    if "train/clip_fraction" in nv:
                        metrics["clip_fraction"] = round(nv["train/clip_fraction"], 4)
                except Exception:
                    pass

            self.socketio.emit("training_metrics", metrics)

        # Save checkpoint
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}")
            self.model.save(model_path)
            logger.info(f"[{self.session_id}] Checkpoint saved at step {self.num_timesteps}")
            self.socketio.emit("training_checkpoint", {
                "session_id": self.session_id,
                "path": model_path,
                "timesteps": self.num_timesteps,
            })

        return True
