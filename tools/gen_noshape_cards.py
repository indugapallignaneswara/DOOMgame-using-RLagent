"""Generate model cards for the 9 no-shape PPO models.

Wrapper around build_model_cards.py that filters registry to
only process ppo_noshape_* entries. Requires:
  1. Models downloaded locally (via /api/wandb/sync or wandb_loader)
  2. W&B access for training curves

Usage:
    python tools/gen_noshape_cards.py --out-dir cards/
    python tools/gen_noshape_cards.py --out-dir cards/ --skip-wandb  # no curve fetch
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "cards"))
    ap.add_argument("--wandb-entity", default="ignaneswara-srm-institute-of-science-and-technology")
    ap.add_argument("--wandb-project", default="doom-rl")
    ap.add_argument("--skip-wandb", action="store_true", help="Skip W&B curve fetch")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()

    # Load registry and filter to no-shape models
    with open(config.REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    noshape = [m for m in registry["models"]
               if m.get("name", "").startswith("ppo_noshape_")]

    if not noshape:
        print("No ppo_noshape_* models found in registry. Run register_noshape.py first.")
        return

    # Check which have local paths
    available = []
    missing = []
    for m in noshape:
        path = m.get("path", "")
        if path and os.path.exists(path):
            available.append(m)
        else:
            missing.append(m)

    if missing:
        print(f"\n{len(missing)} models need to be downloaded from W&B first:")
        for m in missing:
            print(f"  - {m['name']} (scenario: {m['scenario']})")
        print("\nDownload them via:")
        print("  POST /api/wandb/sync  (from the web UI)")
        print("  or: python -c \"from doom.wandb_loader import sync_all_wandb_models; sync_all_wandb_models()\"")

    if not available:
        print("\nNo models available locally. Download from W&B first.")
        return

    print(f"\nGenerating cards for {len(available)} available no-shape models...")
    os.makedirs(args.out_dir, exist_ok=True)

    # Create a temporary registry with only the available no-shape models
    temp_registry = {"models": available}
    temp_path = os.path.join(args.out_dir, "_noshape_registry.json")
    with open(temp_path, "w") as f:
        json.dump(temp_registry, f, indent=2)

    # Build the command to run build_model_cards.py
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "tools", "build_model_cards.py"),
        "--registry", temp_path,
        "--out-dir", args.out_dir,
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
    ]
    if not args.skip_wandb:
        cmd.extend(["--wandb-entity", args.wandb_entity,
                     "--wandb-project", args.wandb_project])

    print(f"Running: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    # Clean up temp registry
    try:
        os.remove(temp_path)
    except OSError:
        pass

    if result.returncode == 0:
        print(f"\nCards generated in {args.out_dir}/")
    else:
        print(f"\nCard generation failed with code {result.returncode}")


if __name__ == "__main__":
    main()
