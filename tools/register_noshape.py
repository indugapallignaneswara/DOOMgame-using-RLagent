"""Register the 9 no-shape PPO models into the local registry.

These models were trained with --no-reward-shaping on the training VM.
They exist as W&B artifacts (ppo_<scenario>_noshape_final) and can be
downloaded via the /api/wandb/download endpoint or doom.wandb_loader.

This script adds registry entries so the UI can list and compare them
even before the .zip files are downloaded locally.
"""

import json
import os
import sys
import uuid
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config

# Final mean rewards from HANDOFF.md training results
NOSHAPE_RESULTS = {
    "basic":              {"mean_reward": 79.9,  "total_timesteps": 100000},
    "defend_the_line":    {"mean_reward": 20.3,  "total_timesteps": 100000},
    "defend_the_center":  {"mean_reward": 12.6,  "total_timesteps": 100000},
    "take_cover":         {"mean_reward": 274.8, "total_timesteps": 100000},
    "health_gathering":   {"mean_reward": 315.7, "total_timesteps": 100000},
    "my_way_home":        {"mean_reward": 0.1,   "total_timesteps": 200000},
    "deadly_corridor_s1": {"mean_reward": 2070.9, "total_timesteps": 400000},
    "deadly_corridor_s3": {"mean_reward": 1210.8, "total_timesteps": 400000},
    "deadly_corridor_s5": {"mean_reward": 223.0,  "total_timesteps": 400000},
}


def main():
    registry_path = config.REGISTRY_PATH
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    existing_names = {m["name"] for m in registry["models"]}
    added = 0

    for scenario, info in NOSHAPE_RESULTS.items():
        name = f"ppo_noshape_{scenario}"
        if name in existing_names:
            print(f"  SKIP  {name} (already registered)")
            continue

        entry = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "scenario": scenario,
            "algo": "ppo",
            "reward_shaping": False,
            "path": "",  # empty until downloaded from W&B
            "wandb_artifact": f"ppo_{scenario}_final",
            "description": f"PPO agent on {scenario} WITHOUT reward shaping.",
            "hyperparams": {
                "learning_rate": 0.0001,
                "n_steps": 2048 if "corridor" not in scenario else 8192,
                "clip_range": 0.2 if "corridor" not in scenario else 0.1,
                "gamma": 0.99 if "corridor" not in scenario else 0.95,
                "gae_lambda": 0.95 if "corridor" not in scenario else 0.9,
            },
            "total_timesteps": info["total_timesteps"],
            "created_at": "2026-05-03T00:00:00",
            "mean_reward": info["mean_reward"],
            "total_episodes": 0,
            "source": "wandb",
        }
        registry["models"].append(entry)
        added += 1
        print(f"  ADD   {name}  (id={entry['id']}, reward={info['mean_reward']})")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nDone. Added {added} no-shape models. Total registry: {len(registry['models'])} models.")


if __name__ == "__main__":
    main()
