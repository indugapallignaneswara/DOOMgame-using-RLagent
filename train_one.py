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
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from doom.scenarios import get_scenario, SCENARIOS
from doom.environment import VizDoomGym
from doom.rewards import RewardShapingWrapper

ALGOS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


def build_env(scenario_key, reward_config, use_shaping: bool = True):
    env = VizDoomGym(scenario_key, render=False)
    if use_shaping and any(v != 0 for v in reward_config.values()):
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


def _build_model(algo: str, env, hp: dict, total_timesteps: int, device: str,
                 seed: int, tb_log: str):
    """Construct an SB3 model for the chosen algorithm.

    Hyperparams come from the scenario's `default_hyperparams` (PPO-shaped).
    For DQN/A2C we map the PPO-style hp dict to algo-appropriate kwargs.
    """
    common = dict(policy="CnnPolicy", env=env, verbose=0, device=device,
                  seed=seed, tensorboard_log=tb_log,
                  learning_rate=hp["learning_rate"])

    if algo == "ppo":
        return PPO(**common,
                   n_steps=hp["n_steps"],
                   clip_range=hp["clip_range"],
                   gamma=hp["gamma"],
                   gae_lambda=hp["gae_lambda"])

    if algo == "a2c":
        # A2C is on-policy like PPO; no clip_range. Use a smaller n_steps
        # since A2C does many small updates instead of PPO's batched epochs.
        return A2C(**common,
                   n_steps=min(hp["n_steps"], 64),
                   gamma=hp["gamma"],
                   gae_lambda=hp["gae_lambda"])

    if algo == "dqn":
        # DQN is off-policy with a replay buffer. Defaults assume a long
        # training run; for our 100K-400K step budgets we shorten the
        # exploration phase and start learning early so the agent has
        # time to actually train.
        learning_starts = min(5000, total_timesteps // 20)
        target_update_interval = max(500, total_timesteps // 100)
        # DQN replay buffer: cap at 100K to avoid eating RAM with 9 parallel
        # runs (each buffer is ~ buffer_size * obs_size * 1 byte = ~1.6 GB
        # for 100K steps at 100x160 uint8).
        buffer_size = min(100_000, total_timesteps)
        return DQN(**common,
                   buffer_size=buffer_size,
                   learning_starts=learning_starts,
                   batch_size=64,
                   tau=1.0,
                   gamma=hp["gamma"],
                   train_freq=4,
                   target_update_interval=target_update_interval,
                   exploration_fraction=0.3,
                   exploration_final_eps=0.05)

    raise ValueError(f"unknown algo: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    ap.add_argument("--algo", default="ppo", choices=list(ALGOS.keys()))
    ap.add_argument("--out-dir", required=True, help="Where to save model + checkpoints")
    ap.add_argument("--wandb-project", default="doom-rl")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="Override scenario default_hyperparams total_timesteps")
    ap.add_argument("--no-reward-shaping", action="store_true",
                    help="Skip the RewardShapingWrapper; use raw env reward only")
    args = ap.parse_args()

    scenario = get_scenario(args.scenario)
    hp = scenario["default_hyperparams"]
    rc = scenario["reward_defaults"]
    total_timesteps = args.total_timesteps or hp["total_timesteps"]

    os.makedirs(args.out_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.algo}_{args.scenario}",
        config={
            "scenario": args.scenario,
            "algo": args.algo.upper(),
            "policy": "CnnPolicy",
            "device": args.device,
            "seed": args.seed,
            "total_timesteps": total_timesteps,
            **hp,
            **{f"reward_{k}": v for k, v in rc.items()},
        },
        sync_tensorboard=True,
        save_code=False,
        reinit=True,
        tags=[args.algo, args.scenario],
    )

    env = build_env(args.scenario, rc, use_shaping=not args.no_reward_shaping)

    tb_log = os.path.join(args.out_dir, "tb")
    model = _build_model(args.algo, env, hp, total_timesteps,
                          args.device, args.seed, tb_log)

    tag = f"{args.algo}_{args.scenario}"
    print(f"[{tag}] device={model.device} total_timesteps={total_timesteps}", flush=True)

    ckpt_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 10000),
        save_path=os.path.join(args.out_dir, "checkpoints"),
        name_prefix=tag,
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
        final_path = os.path.join(args.out_dir, f"{tag}_final.zip")
        model.save(final_path)
        wandb.log({"train/wall_seconds": time.time() - t0})
        # Upload final model as W&B artifact
        art = wandb.Artifact(name=f"{tag}_final", type="model",
                              metadata={"scenario": args.scenario,
                                        "algo": args.algo, **hp})
        art.add_file(final_path)
        run.log_artifact(art)
        print(f"[{tag}] DONE in {time.time()-t0:.1f}s -> {final_path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        run.finish()


if __name__ == "__main__":
    main()
