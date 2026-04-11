"""Training manager - handles lifecycle of training sessions."""

import os
import json
import time
import logging
import threading
import uuid
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from doom.environment import VizDoomGym
from doom.rewards import RewardShapingWrapper
from doom.callbacks import WebTrainingCallback
import config

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages concurrent training sessions with background threads."""

    def __init__(self, socketio):
        self.socketio = socketio
        self.active_sessions = {}
        self._lock = threading.Lock()

    def start_training(self, session_id, scenario_key, hyperparams,
                       reward_config, total_timesteps):
        """Launch training in a background thread."""
        pause_event = threading.Event()
        pause_event.set()  # Start unpaused
        stop_flag = threading.Event()

        save_path = os.path.join(config.TRAIN_DIR, f"session_{session_id}")
        os.makedirs(save_path, exist_ok=True)

        with self._lock:
            self.active_sessions[session_id] = {
                "status": "starting",
                "scenario": scenario_key,
                "hyperparams": hyperparams,
                "reward_config": reward_config,
                "total_timesteps": total_timesteps,
                "pause_event": pause_event,
                "stop_flag": stop_flag,
                "started_at": datetime.now().isoformat(),
                "thread": None,
            }

        thread = threading.Thread(
            target=self._train_worker,
            args=(session_id, scenario_key, hyperparams, reward_config,
                  total_timesteps, save_path, pause_event, stop_flag),
            daemon=True,
        )
        thread.start()

        with self._lock:
            self.active_sessions[session_id]["thread"] = thread
            self.active_sessions[session_id]["status"] = "running"

    def _train_worker(self, session_id, scenario_key, hyperparams,
                      reward_config, total_timesteps, save_path,
                      pause_event, stop_flag):
        """Background worker that runs PPO training."""
        env = None
        try:
            logger.info(f"[{session_id}] Creating environment: {scenario_key}")
            self.socketio.emit("training_status", {
                "session_id": session_id, "status": "initializing",
                "message": "Creating ViZDoom environment..."
            })
            env = VizDoomGym(scenario_key, render=False)

            # Apply reward shaping if configured
            has_shaping = any(v != 0 for v in reward_config.values())
            if has_shaping:
                env = RewardShapingWrapper(env, **reward_config)

            env = Monitor(env)

            self.socketio.emit("training_status", {
                "session_id": session_id, "status": "initializing",
                "message": "Building PPO model..."
            })

            # Create PPO model
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=hyperparams.get("learning_rate", 0.0001),
                n_steps=hyperparams.get("n_steps", 2048),
                clip_range=hyperparams.get("clip_range", 0.2),
                gamma=hyperparams.get("gamma", 0.99),
                gae_lambda=hyperparams.get("gae_lambda", 0.95),
                verbose=0,
                tensorboard_log=os.path.join(config.LOGS_DIR, f"session_{session_id}"),
            )

            callback = WebTrainingCallback(
                socketio=self.socketio,
                session_id=session_id,
                total_timesteps=total_timesteps,
                emit_freq=50,
                check_freq=10000,
                save_path=save_path,
                pause_event=pause_event,
                stop_flag=stop_flag,
            )

            logger.info(f"[{session_id}] Training started: {total_timesteps} timesteps")
            self.socketio.emit("training_status", {
                "session_id": session_id, "status": "training"
            })

            model.learn(total_timesteps=total_timesteps, callback=callback)

            # Save final model
            final_path = os.path.join(save_path, "final_model")
            model.save(final_path)
            logger.info(f"[{session_id}] Final model saved: {final_path}")

            # Register model
            self._register_model(session_id, scenario_key, hyperparams,
                                 reward_config, final_path + ".zip",
                                 total_timesteps, callback._episode_rewards)

            with self._lock:
                self.active_sessions[session_id]["status"] = "completed"

            # Update training history
            self._update_history(session_id, "completed", callback)

            self.socketio.emit("training_status", {
                "session_id": session_id,
                "status": "completed",
                "model_path": final_path,
            })

        except Exception as e:
            logger.error(f"[{session_id}] Training failed: {e}")
            with self._lock:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["status"] = "failed"
            self._update_history(session_id, "failed", error=str(e))
            self.socketio.emit("training_status", {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
            })
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _register_model(self, session_id, scenario_key, hyperparams,
                        reward_config, path, timesteps, episode_rewards):
        """Add trained model to the registry."""
        model_entry = {
            "id": str(uuid.uuid4())[:8],
            "name": f"{scenario_key}_{session_id}",
            "scenario": scenario_key,
            "path": path,
            "hyperparams": hyperparams,
            "reward_config": reward_config,
            "total_timesteps": timesteps,
            "created_at": datetime.now().isoformat(),
        }

        if episode_rewards:
            model_entry["mean_reward"] = round(
                sum(episode_rewards[-50:]) / max(len(episode_rewards[-50:]), 1), 2
            )
            model_entry["total_episodes"] = len(episode_rewards)

        try:
            registry_path = config.REGISTRY_PATH
            if os.path.exists(registry_path):
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            else:
                registry = {"models": []}

            registry["models"].append(model_entry)

            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            logger.info(f"Model registered: {model_entry['id']}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    def _update_history(self, session_id, status, callback=None, error=None):
        """Update training history with final results."""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from app import save_training_session, load_history
            from datetime import datetime

            history = load_history()
            for s in history["sessions"]:
                if s["session_id"] == session_id:
                    s["status"] = status
                    s["finished_at"] = datetime.now().isoformat()
                    if callback and callback._episode_rewards:
                        last50 = callback._episode_rewards[-50:]
                        s["mean_reward"] = round(sum(last50) / len(last50), 2)
                        s["total_episodes"] = len(callback._episode_rewards)
                    if error:
                        s["error"] = error
                    save_training_session(s)
                    break
        except Exception as e:
            logger.error(f"Failed to update history: {e}")

    def pause_training(self, session_id):
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["pause_event"].clear()
                self.active_sessions[session_id]["status"] = "paused"

    def resume_training(self, session_id):
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["pause_event"].set()
                self.active_sessions[session_id]["status"] = "running"

    def stop_training(self, session_id):
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["stop_flag"].set()
                self.active_sessions[session_id]["pause_event"].set()  # unblock if paused
                self.active_sessions[session_id]["status"] = "stopping"

    def get_status(self, session_id):
        with self._lock:
            if session_id in self.active_sessions:
                return self.active_sessions[session_id].get("status", "unknown")
        return "not_found"
