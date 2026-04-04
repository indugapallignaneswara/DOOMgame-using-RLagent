"""Evaluation engine for trained models."""

import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from doom.environment import VizDoomGym
from doom.rewards import RewardShapingWrapper

logger = logging.getLogger(__name__)


class Evaluator:
    """Run evaluation episodes and collect statistics."""

    def evaluate(self, model_path, scenario_key, n_episodes=10, reward_config=None):
        """Evaluate a model over n_episodes and return statistics.

        Returns:
            dict with mean_reward, std_reward, min, max, rewards_list, episode_lengths
        """
        env = None
        try:
            logger.info(f"Evaluating {model_path} on {scenario_key} for {n_episodes} episodes")

            model = PPO.load(model_path)
            env = VizDoomGym(scenario_key, render=False)

            if reward_config and any(v != 0 for v in reward_config.values()):
                env = RewardShapingWrapper(env, **reward_config)

            env = Monitor(env)

            rewards = []
            lengths = []

            for ep in range(n_episodes):
                obs, info = env.reset()
                total_reward = 0
                steps = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1

                rewards.append(float(total_reward))
                lengths.append(steps)
                logger.info(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.1f}, steps={steps}")

            results = {
                "scenario": scenario_key,
                "n_episodes": n_episodes,
                "mean_reward": round(float(np.mean(rewards)), 2),
                "std_reward": round(float(np.std(rewards)), 2),
                "min_reward": round(float(np.min(rewards)), 2),
                "max_reward": round(float(np.max(rewards)), 2),
                "rewards_list": [round(r, 2) for r in rewards],
                "episode_lengths": lengths,
                "mean_length": round(float(np.mean(lengths)), 1),
            }

            logger.info(f"Evaluation complete: mean={results['mean_reward']}, std={results['std_reward']}")
            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
