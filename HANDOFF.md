# Handoff — DOOM RL Arena (state as of 2026-05-03)

This document is the bridge between this VM and any future one. It tells the
next session (a) where every artefact lives, (b) what's already trained, (c)
how to resume work without redoing it.

## What's been built

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full plan and the five
moonshots. Of those, the Brain-cam saliency overlay, model card generator,
and Belief Inspector first pass are shipped on `parallel-training-and-arch-plan`.

| Component | Files | Status |
|---|---|---|
| Parallel multi-algo training driver | `train_one.py`, `train_all.py` | shipped — 30 models trained |
| Perturbation saliency module | `doom/saliency.py` | shipped — passes Adebayo randomisation test |
| Saliency CLI tool (MP4 + strip) | `tools/saliency_demo.py` | shipped |
| Brain-cam UI (live `/play` overlay) | `doom/player.py`, `templates/play.html`, `static/js/player.js` | shipped |
| Belief Inspector (probs + V(s) + prediction widget) | same files + `static/css/styles.css` | shipped |
| Model card generator + gallery | `tools/build_model_cards.py` | shipped — 21 cards generated |

## What's been trained

30 PPO/DQN/A2C agents across 9 ViZDoom scenarios. Final mean episodic reward:

| Scenario | PPO (shaped) | DQN | A2C | PPO (no-shape) | shaping Δ |
|---|---:|---:|---:|---:|---:|
| basic | 80.2 | 71.1 | −300.8 | 79.9 | +0.3 |
| defend_the_line | 17.3 | 16.5 | — | 20.3 | −3.0 |
| defend_the_center | 9.6 | 4.8 | 0.8 | 12.6 | −3.0 |
| take_cover | 356.3 | 337.7 | 226.9 | 274.8 | +81.5 |
| health_gathering | 287.8 | 413.3 | — | 315.7 | −27.9 |
| my_way_home | −2.6 | −2.6 | — | 0.1 | −2.7 |
| deadly_corridor_s1 | 2042.4 | 1732.1 | — | 2070.9 | −28.5 |
| deadly_corridor_s3 | 1251.8 | 935.0 | — | 1210.8 | +41.0 |
| deadly_corridor_s5 | 228.6 | 185.9 | — | 223.0 | +5.6 |

Highlights worth remembering:

- **A2C diverges spectacularly on `basic` (−300.8)** — collapse where PPO and DQN succeed. A pedagogical asset, not a bug.
- **DQN beats PPO on `health_gathering` (+125)** — replay buffer wins on sparse-reward survival.
- **PPO `deadly_corridor_s1` reward-hacks** — fires ATTACK 0% of the time in deterministic eval, just sprints MOVE_FORWARD. The 2042 reward is entirely from navigation/reach-vest signals, not kills. Centerpiece example for a "your reward is not what you think" lesson.
- **Reward shaping is NOT a free lunch** — helps on 2–3 of 9 scenarios, hurts on 4–5. Strongest negative effect: `health_gathering` (−27.9) and `deadly_corridor_s1` (−28.5). Strongest positive: `take_cover` (+81.5).

## Where the artefacts live

Persistent disk (`/mnt/orbit_war_disk`, survives VM destroy):

```
/mnt/orbit_war_disk/
├── BOOTSTRAP.md                            ← VM bootstrap (updated for doom-rl)
├── doom-checkpoints/                       ← 13 GB
│   ├── <scenario>/<scenario>_final.zip     ← original PPO (×9)
│   ├── <scenario>/dqn/dqn_<scenario>_final.zip
│   ├── <scenario>/a2c/a2c_<scenario>_final.zip
│   ├── <scenario>/ppo_noshape/ppo_<scenario>_final.zip
│   ├── cards/<algo>_<scenario>.png         ← 21 model cards
│   ├── cards/index.html                    ← gallery
│   └── saliency-demos/<scenario>/          ← MP4s + strip.png
├── doom-handoff/                           ← this handoff
│   ├── logs/                               ← all training logs (41 MB)
│   └── registry-snapshot/registry.json     ← snapshot of the 21+9 model paths
└── venv/                                   ← orbit-war's persistent torch+ROCm venv
```

**W&B project:** `https://wandb.ai/ignaneswara-srm-institute-of-science-and-technology/doom-rl` — every run is logged with mean reward, ep_len, loss curves, and a final-model artefact. Run names: `<algo>_<scenario>` (e.g. `dqn_basic`); the original 9 PPO runs are named bare `<scenario>` (`basic`, etc.).

**GitHub:** branch `parallel-training-and-arch-plan` carries every code change.

## Resuming on a fresh VM

```bash
# 1. The persistent disk auto-mounts at /mnt/orbit_war_disk. If not:
mount /dev/sda /mnt/orbit_war_disk

# 2. Recreate the doom-venv from the orbit-war one (5 min, no fresh download):
cp -a /mnt/orbit_war_disk/venv /root/doom-venv
/root/doom-venv/bin/python -m venv --upgrade /root/doom-venv

# 3. Re-anchor pip's launcher (the cp leaves stale shebangs):
/root/doom-venv/bin/python -m pip install --no-input \
    vizdoom stable-baselines3 'gymnasium[classic-control]' \
    opencv-python-headless wandb tensorboard scipy imageio imageio-ffmpeg \
    flask flask-socketio simple-websocket
# (most are no-ops — re-runs are idempotent)

# 4. Clone the repo:
git clone https://github.com/indugapallignaneswara/DOOMgame-using-RLagent.git /root/doom/DOOMgame-using-RLagent
cd /root/doom/DOOMgame-using-RLagent
git checkout parallel-training-and-arch-plan   # if work isn't yet on main

# 5. Restore registry.json (paths are absolute, host-specific):
mkdir -p models
cp /mnt/orbit_war_disk/doom-handoff/registry-snapshot/registry.json models/registry.json

# 6. Authenticate W&B (if /root/.netrc is gone):
WANDB_API_KEY=<your key> /root/doom-venv/bin/python -m wandb login --relogin

# 7. Run the web app:
mkdir -p /root/doom/logs
nohup /root/doom-venv/bin/python app.py > /root/doom/logs/web.log 2>&1 &
```

`http://localhost:5000/play` should now load. Pick any of the 21 registered
models, toggle Brain-cam to "Policy attention," toggle Prediction mode, and
the saliency overlay + action bars + V(s) sparkline + prediction widget all
work end-to-end against pre-trained weights.

## Important gotchas (learned the hard way)

- **MIOpen race on AMD MI300X.** ROCm's MIOpen kernel-find phase serialises poorly when many processes start a CnnPolicy forward at once. We hit `miopenStatusInternalError` on 9 parallel PPO launches. Fix: rely on MIOpen defaults (`~/.config/miopen` for the SQLite kernel DB, `~/.cache/miopen` for compiled binaries). Pointing both `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` at the same directory creates a folder named `gfx942.ukdb` that later can't be opened as a SQLite file. The launch stagger in `train_all.py` (90 s for the first proc, 8 s for the rest) handles cold compile.

- **Tmux sessions get cleaned up between bash invocations.** A long-running training driver started under `tmux new -d -s ...` from one bash call can be killed when later bash calls happen. Use `setsid nohup <cmd> & disown` to fully orphan the process from the parent shell. `train_all.py` then survives.

- **SB3 stores `observation_space` in CHW after a load**, while the env yields HWC. Saliency / belief code needs to detect both: `if shape[0] in (1, 3, 4): C, H, W = shape else: H, W = shape[:2]`.

- **`models/registry.json` paths are absolute** (`/mnt/orbit_war_disk/...`). Do NOT commit it — restore from `/mnt/orbit_war_disk/doom-handoff/registry-snapshot/` on each VM.

- **Grad-CAM is interpretability theatre for RL.** We use perturbation saliency (Greydanus 2018) instead, with Adebayo et al. 2018 sanity check baked in. See `doom/saliency.py:235` for the randomisation test.

## Open work (next steps)

In priority order, per the reviewer synthesis in `ARCHITECTURE.md` §9:

1. **Lesson templates** ("PPO vs DQN vs A2C side-by-side", "with vs without shaping", "γ ablation"). Data is already in W&B; needs a UI page that pulls runs by tag and renders side-by-side curves + side-by-side gameplay.
2. **Register the 9 no-shape models** — they're trained to disk but not added to `models/registry.json`. Just rerun the registry-update script (snippet in `tools/`).
3. **Generate cards for the 9 no-shape models** — `python tools/build_model_cards.py` will do all of registry; either re-add the no-shape entries first or pass `--only <ids>`.
4. **Phase 0 hygiene fixes** — `model_path` sandboxing, hyperparam validation, JPEG encoding off the env-step thread.
5. **Phase 2 foundation rebuild** — FastAPI + Postgres + Redis + Ray + WebRTC, when the project is ready to scale beyond a single-VM proof of concept.

## Provenance

- 30 trained checkpoints, 21 of them registered, 9 awaiting registration.
- Code on `parallel-training-and-arch-plan`; main is at the same content modulo three squashed/rebased SHAs.
- This handoff is hand-written and current as of the last training run; treat it as authoritative when conflicting with older notes.
