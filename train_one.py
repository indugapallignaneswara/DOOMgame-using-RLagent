"""Train a single PPO agent on one ViZDoom scenario, log to W&B, save checkpoints.

Run as subprocess from train_all.py — one process per scenario so each gets its own
ViZDoom C engine (the bottleneck for parallel training is CPU env stepping, not the
tiny CnnPolicy on the GPU).
"""

import argparse
import os
import sys
import time

# Ensure the repo root is on sys.path so we can `from doom.* import ...`
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from doom.scenarios import get_scenario, SCENARIOS
from doom.environment import VizDoomGym
from doom.rewards import RewardShapingWrapper


def build_env(scenario_key, reward_config):
    env = VizDoomGym(scenario_key, render=False)
    if any(v != 0 for v in reward_config.values()):
        env = RewardShapingWrapper(env, **reward_config)
    env = Monitor(env)
    return env


class EpisodeRewardLogger(BaseCallback):
    """Log mean episodic reward to W&B from the Monitor buffer."""
    def __init__(self, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq < self.training_env.num_envs:
            buf = self.model.ep_info_buffer
            if buf and len(buf) > 0:
                mean_r = sum(ep["r"] for ep in buf) / len(buf)
                mean_l = sum(ep["l"] for ep in buf) / len(buf)
                wandb.log({
                    "rollout/ep_rew_mean": mean_r,
                    "rollout/ep_len_mean": mean_l,
                    "global_step": self.num_timesteps,
                })
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    ap.add_argument("--out-dir", required=True, help="Where to save model + checkpoints")
    ap.add_argument("--wandb-project", default="doom-rl")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenario = get_scenario(args.scenario)
    hp = scenario["default_hyperparams"]
    rc = scenario["reward_defaults"]
    total_timesteps = hp["total_timesteps"]

    os.makedirs(args.out_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.scenario}",
        config={
            "scenario": args.scenario,
            "algo": "PPO",
            "policy": "CnnPolicy",
            "device": args.device,
            "seed": args.seed,
            **hp,
            **{f"reward_{k}": v for k, v in rc.items()},
        },
        sync_tensorboard=True,
        save_code=False,
        reinit=True,
    )

    env = build_env(args.scenario, rc)

    tb_log = os.path.join(args.out_dir, "tb")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        clip_range=hp["clip_range"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        verbose=0,
        device=args.device,
        seed=args.seed,
        tensorboard_log=tb_log,
    )

    print(f"[{args.scenario}] device={model.device} total_timesteps={total_timesteps}", flush=True)

    ckpt_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 10000),
        save_path=os.path.join(args.out_dir, "checkpoints"),
        name_prefix=args.scenario,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    wandb_cb = WandbCallback(
        model_save_path=os.path.join(args.out_dir, "wandb_models"),
        model_save_freq=max(total_timesteps // 5, 10000),
        gradient_save_freq=0,
        verbose=0,
    )
    rew_cb = EpisodeRewardLogger(log_freq=2048)

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([ckpt_cb, wandb_cb, rew_cb]),
            progress_bar=False,
        )
        final_path = os.path.join(args.out_dir, f"{args.scenario}_final.zip")
        model.save(final_path)
        wandb.log({"train/wall_seconds": time.time() - t0})
        # Upload final model as W&B artifact
        art = wandb.Artifact(name=f"{args.scenario}_final", type="model",
                              metadata={"scenario": args.scenario, **hp})
        art.add_file(final_path)
        run.log_artifact(art)
        print(f"[{args.scenario}] DONE in {time.time()-t0:.1f}s -> {final_path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        run.finish()


if __name__ == "__main__":
    main()
