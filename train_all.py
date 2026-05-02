"""Launch parallel PPO trainings, one subprocess per scenario.

Each subprocess gets its own ViZDoom C engine. All share the GPU for the
tiny CnnPolicy. With 9 scenarios on a 20-core box, each run ~2 cores.
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from doom.scenarios import SCENARIOS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/mnt/orbit_war_disk/doom-checkpoints")
    ap.add_argument("--log-dir", default="/root/doom/logs")
    ap.add_argument("--wandb-project", default="doom-rl")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--python", default="/root/doom-venv/bin/python")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Subset of scenarios; default = all")
    ap.add_argument("--algos", nargs="*", default=["ppo"],
                    help="Algorithms to run on each scenario (ppo|dqn|a2c)")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="Override the per-scenario default_hyperparams.total_timesteps")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    scenarios = args.scenarios or list(SCENARIOS.keys())

    # Set MIOpen / hip caches to per-process subdirs to avoid contention
    base_env = os.environ.copy()
    base_env["WANDB_DIR"] = os.path.join(args.log_dir, "wandb")
    base_env["WANDB_SILENT"] = "true"
    base_env["TOKENIZERS_PARALLELISM"] = "false"
    # ViZDoom is single-threaded; we want each torch process to use few threads
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    os.makedirs(base_env["WANDB_DIR"], exist_ok=True)

    # Build the (algo, scenario) cartesian product to launch.
    jobs = [(algo, s) for algo in args.algos for s in scenarios]

    procs = {}
    for algo, s in jobs:
        tag = f"{algo}_{s}"
        out_dir = os.path.join(args.out_root, s, algo)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"{tag}.log")
        log_fh = open(log_path, "w", buffering=1)

        cmd = [
            args.python, "-u",
            os.path.join(REPO_ROOT, "train_one.py"),
            "--scenario", s,
            "--algo", algo,
            "--out-dir", out_dir,
            "--wandb-project", args.wandb_project,
        ]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
        if args.total_timesteps:
            cmd += ["--total-timesteps", str(args.total_timesteps)]

        env = base_env.copy()
        # Use MIOpen's defaults: ~/.config/miopen for the SQLite kernel DB,
        # ~/.cache/miopen for compiled kernel binaries. The orbit-war training
        # already uses these successfully. Pointing both env vars at the same
        # directory makes MIOpen create a `gfx942.ukdb` *folder* for kernel
        # binaries, which then can't be opened later as a SQLite file.
        # Just rely on the launch stagger so the first proc compiles before
        # the rest start.

        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)
        procs[tag] = (p, log_fh, log_path)
        print(f"launched {tag} pid={p.pid} -> {log_path}", flush=True)
        # Long stagger ONLY for the first process (which does the cold compile);
        # after that, kernels are cached so subsequent procs can start fast.
        time.sleep(90 if len(procs) == 1 else 8)

    # Poll until everyone is done; print any death immediately
    pending = set(procs.keys())
    while pending:
        time.sleep(15)
        for tag in list(pending):
            p, _, log_path = procs[tag]
            rc = p.poll()
            if rc is not None:
                pending.discard(tag)
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"[done] {tag}: {status} (log: {log_path})", flush=True)

    # Summary
    print("\n=== summary ===")
    bad = 0
    for tag, (p, _, log_path) in procs.items():
        rc = p.returncode
        print(f"  {tag}: rc={rc}  log={log_path}")
        if rc != 0:
            bad += 1
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
