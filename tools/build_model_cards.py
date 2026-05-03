"""Generate shareable model cards from the registry.

Per ARCHITECTURE.md §4.5 and Park's review (§9): every saved model gets a
1200x630 PNG (LinkedIn/Twitter open-graph aspect ratio) summarising its
training and personality. This is the *acquisition feature* — the
Spotify-Wrapped of an RL agent: legible to recruiters, postable on LinkedIn,
screenshottable on X.

Output
------
  /mnt/orbit_war_disk/doom-checkpoints/cards/<algo>_<scenario>.png
  /mnt/orbit_war_disk/doom-checkpoints/cards/index.html

Usage
-----
    python tools/build_model_cards.py \\
        --registry models/registry.json \\
        --out-dir /mnt/orbit_war_disk/doom-checkpoints/cards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN, A2C

from doom.environment import VizDoomGym
from doom.scenarios import get_scenario


# --- theme ------------------------------------------------------------------

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
DIM = "#8b949e"
MAGENTA = "#ff3ec9"
CYAN = "#39d9ff"
AMBER = "#f0b941"
GRID = "#262d36"

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

CARD_W, CARD_H = 1200, 630
DPI = 100  # 12in x 6.3in @ 100 dpi -> 1200x630


# --- data classes -----------------------------------------------------------

@dataclass
class Curve:
    steps: np.ndarray
    rewards: np.ndarray
    has_data: bool


@dataclass
class Personality:
    aggression: float | None  # None for scenarios without ATTACK
    exploration: float
    efficiency: float  # raw value; normalised at card-build time
    total_reward: float
    steps: int


@dataclass
class CardData:
    model_id: str
    name: str
    algo: str
    scenario: str
    timesteps: int
    mean_reward: float
    wall_seconds: float | None
    curve: Curve
    personality: Personality


# --- W&B curve fetch --------------------------------------------------------

def fetch_curve(api: Any, entity: str, project: str, run_name: str,
                fallback_names: list[str] | None = None) -> tuple[Curve, float | None]:
    """Pull rollout/ep_rew_mean from the named run; returns (curve, wall_s).

    Tries `run_name` first, then each `fallback_names` in order. Necessary
    because the original 9 PPO runs were named bare `<scenario>` (e.g. "basic"),
    while the multi-algo runs are named `<algo>_<scenario>` ("dqn_basic").
    """
    candidates = [run_name] + (fallback_names or [])
    runs = []
    for name in candidates:
        hits = list(api.runs(f"{entity}/{project}",
                              filters={"display_name": name},
                              order="-created_at"))
        if hits:
            runs = hits
            break
    if not runs:
        return Curve(np.array([]), np.array([]), False), None

    run = runs[0]
    keys = ["rollout/ep_rew_mean", "global_step", "_step"]
    history = run.history(keys=keys, samples=2000, pandas=False)
    steps, rewards = [], []
    for row in history:
        r = row.get("rollout/ep_rew_mean")
        s = row.get("global_step", row.get("_step"))
        if r is None or s is None:
            continue
        steps.append(float(s))
        rewards.append(float(r))

    wall = run.summary.get("train/wall_seconds") or run.summary.get("_runtime")
    wall = float(wall) if wall is not None else None

    if len(steps) < 3:
        return Curve(np.array(steps), np.array(rewards), False), wall

    order = np.argsort(steps)
    return Curve(np.array(steps)[order], np.array(rewards)[order], True), wall


def ema(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# --- personality probe ------------------------------------------------------

def probe_personality(model_path: str, algo: str, scenario: str,
                       n_episodes: int = 3, max_steps: int = 1500) -> Personality:
    """Roll deterministic episodes and compute aggression/exploration/efficiency."""
    cls = ALGO_CLASSES[algo]
    model = cls.load(model_path, device="cpu")

    env = VizDoomGym(scenario, render=False)
    buttons = get_scenario(scenario)["buttons"]
    attack_idx = buttons.index("ATTACK") if "ATTACK" in buttons else None
    n_actions = env.num_actions

    actions: list[int] = []
    total_reward = 0.0
    total_steps = 0
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                a = int(action)
                actions.append(a)
                obs, reward, terminated, truncated, _ = env.step(a)
                total_reward += float(reward)
                total_steps += 1
                if terminated or truncated:
                    break
    finally:
        env.close()

    arr = np.asarray(actions, dtype=np.int64)
    aggression: float | None
    if attack_idx is None or len(arr) == 0:
        aggression = None
    else:
        aggression = float((arr == attack_idx).mean())

    if len(arr) == 0:
        exploration = 0.0
    else:
        counts = np.bincount(arr, minlength=n_actions).astype(float)
        probs = counts / counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(probs > 0, probs * np.log(probs), 0.0))
        exploration = float(ent / np.log(n_actions)) if n_actions > 1 else 0.0

    efficiency = total_reward / max(total_steps, 1)
    return Personality(aggression, exploration, efficiency, total_reward, total_steps)


# --- card rendering ---------------------------------------------------------

def _short_id(s: str) -> str:
    return s[:8]


def _algo_color(algo: str) -> str:
    return {"ppo": MAGENTA, "dqn": CYAN, "a2c": AMBER}.get(algo, MAGENTA)


def _bar(ax: plt.Axes, y: float, value: float, label: str,
          color: str, na: bool = False) -> None:
    """Horizontal bar in [0,1] with label on the left and value on the right."""
    ax.add_patch(plt.Rectangle((0.30, y - 0.018), 0.55, 0.036,
                                color=GRID, transform=ax.transAxes, zorder=1))
    if not na:
        v = float(np.clip(value, 0.0, 1.0))
        ax.add_patch(plt.Rectangle((0.30, y - 0.018), 0.55 * v, 0.036,
                                    color=color, transform=ax.transAxes, zorder=2))
    ax.text(0.27, y, label, transform=ax.transAxes, color=DIM,
            fontsize=13, ha="right", va="center", family="DejaVu Sans")
    txt = "N/A" if na else f"{value:.2f}"
    ax.text(0.87, y, txt, transform=ax.transAxes, color=TEXT,
            fontsize=13, ha="left", va="center", family="DejaVu Sans Mono")


def render_card(data: CardData, eff_norm: float, out_path: str) -> None:
    fig = plt.figure(figsize=(CARD_W / DPI, CARD_H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    accent = _algo_color(data.algo)

    # ---- header strip ----
    head = fig.add_axes([0, 0.86, 1, 0.14])
    head.set_facecolor(PANEL)
    head.set_xticks([]); head.set_yticks([])
    for spine in head.spines.values():
        spine.set_visible(False)
    head.text(0.025, 0.55, data.scenario.upper().replace("_", " "),
              color=TEXT, fontsize=26, fontweight="bold",
              family="DejaVu Sans", transform=head.transAxes, va="center")
    head.text(0.025, 0.18, f"DOOM RL ARENA  ·  {data.name}",
              color=DIM, fontsize=11, family="DejaVu Sans Mono",
              transform=head.transAxes, va="center")
    head.text(0.975, 0.55, data.algo.upper(),
              color=accent, fontsize=26, fontweight="bold",
              family="DejaVu Sans Mono", transform=head.transAxes,
              va="center", ha="right")
    head.text(0.975, 0.18, f"{data.timesteps:,} steps",
              color=DIM, fontsize=11, family="DejaVu Sans Mono",
              transform=head.transAxes, va="center", ha="right")

    # ---- big number (mean reward) ----
    big = fig.add_axes([0.03, 0.50, 0.42, 0.32])
    big.set_facecolor(BG)
    big.set_xticks([]); big.set_yticks([])
    for spine in big.spines.values():
        spine.set_visible(False)
    big.text(0.0, 0.85, "MEAN EPISODIC REWARD", color=DIM, fontsize=11,
             family="DejaVu Sans Mono", transform=big.transAxes)
    big.text(0.0, 0.30, f"{data.mean_reward:,.1f}", color=TEXT, fontsize=68,
             fontweight="bold", family="DejaVu Sans Mono",
             transform=big.transAxes)

    # ---- training curve thumbnail ----
    curve = fig.add_axes([0.50, 0.50, 0.47, 0.32])
    curve.set_facecolor(BG)
    for spine in curve.spines.values():
        spine.set_visible(False)
    curve.set_xticks([]); curve.set_yticks([])
    if data.curve.has_data:
        smoothed = ema(data.curve.rewards, alpha=0.12)
        curve.plot(data.curve.steps, data.curve.rewards,
                   color=accent, alpha=0.18, linewidth=1.2)
        curve.plot(data.curve.steps, smoothed, color=accent,
                   linewidth=2.6, solid_capstyle="round")
        # subtle gradient fill under the smoothed line
        ymin = float(min(smoothed.min(), data.curve.rewards.min()))
        curve.fill_between(data.curve.steps, ymin, smoothed,
                           color=accent, alpha=0.07)
        curve.text(0.0, 1.02, "TRAINING CURVE", color=DIM, fontsize=11,
                   family="DejaVu Sans Mono", transform=curve.transAxes)
    else:
        curve.text(0.5, 0.5, "training in progress", color=DIM,
                   fontsize=14, family="DejaVu Sans Mono",
                   transform=curve.transAxes, ha="center", va="center",
                   style="italic")
        curve.text(0.0, 1.02, "TRAINING CURVE", color=DIM, fontsize=11,
                   family="DejaVu Sans Mono", transform=curve.transAxes)

    # ---- stats row (wall clock, timesteps) ----
    stats = fig.add_axes([0.03, 0.36, 0.94, 0.10])
    stats.set_facecolor(BG); stats.set_xticks([]); stats.set_yticks([])
    for spine in stats.spines.values():
        spine.set_visible(False)
    if data.wall_seconds is not None:
        wc = _format_duration(data.wall_seconds)
    else:
        wc = "--:--"
    stats.text(0.0, 0.6, "WALL CLOCK", color=DIM, fontsize=11,
               family="DejaVu Sans Mono", transform=stats.transAxes)
    stats.text(0.0, 0.05, wc, color=TEXT, fontsize=22,
               family="DejaVu Sans Mono", transform=stats.transAxes,
               fontweight="bold")
    stats.text(0.22, 0.6, "TIMESTEPS", color=DIM, fontsize=11,
               family="DejaVu Sans Mono", transform=stats.transAxes)
    stats.text(0.22, 0.05, f"{data.timesteps:,}", color=TEXT, fontsize=22,
               family="DejaVu Sans Mono", transform=stats.transAxes,
               fontweight="bold")
    stats.text(0.45, 0.6, "ROLLOUT REWARD", color=DIM, fontsize=11,
               family="DejaVu Sans Mono", transform=stats.transAxes)
    stats.text(0.45, 0.05, f"{data.personality.total_reward:,.1f}",
               color=TEXT, fontsize=22, family="DejaVu Sans Mono",
               transform=stats.transAxes, fontweight="bold")

    # ---- personality scores ----
    pers = fig.add_axes([0.03, 0.10, 0.94, 0.22])
    pers.set_facecolor(BG); pers.set_xticks([]); pers.set_yticks([])
    pers.set_xlim(0, 1); pers.set_ylim(0, 1)
    for spine in pers.spines.values():
        spine.set_visible(False)
    pers.text(0.0, 0.92, "PERSONALITY", color=DIM, fontsize=11,
              family="DejaVu Sans Mono", transform=pers.transAxes)
    p = data.personality
    _bar(pers, 0.68, p.aggression or 0.0, "Aggression",
         MAGENTA, na=p.aggression is None)
    _bar(pers, 0.43, p.exploration, "Exploration", CYAN)
    _bar(pers, 0.18, eff_norm, "Efficiency", AMBER)

    # ---- footer ----
    foot = fig.add_axes([0, 0, 1, 0.06])
    foot.set_facecolor(PANEL); foot.set_xticks([]); foot.set_yticks([])
    for spine in foot.spines.values():
        spine.set_visible(False)
    foot.text(0.025, 0.5, "doom-rl-arena.app", color=DIM, fontsize=11,
              family="DejaVu Sans Mono", transform=foot.transAxes, va="center")
    foot.text(0.975, 0.5, f"id  {_short_id(data.model_id)}", color=DIM,
              fontsize=11, family="DejaVu Sans Mono",
              transform=foot.transAxes, va="center", ha="right")

    fig.savefig(out_path, dpi=DPI, facecolor=BG, edgecolor="none")
    plt.close(fig)


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# --- gallery ----------------------------------------------------------------

GALLERY_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>DOOM RL Arena — Model Cards</title>
<style>
  :root {{ --bg: #0d1117; --panel: #161b22; --text: #e6edf3; --dim: #8b949e;
           --magenta: #ff3ec9; --cyan: #39d9ff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 48px 32px; background: var(--bg);
         color: var(--text); font-family: -apple-system, BlinkMacSystemFont,
         "Segoe UI", Roboto, sans-serif; }}
  header {{ max-width: 1400px; margin: 0 auto 40px; }}
  h1 {{ margin: 0 0 8px; font-size: 38px; letter-spacing: -0.5px; }}
  .sub {{ color: var(--dim); font-family: ui-monospace, SFMono-Regular,
          Menlo, monospace; font-size: 14px; }}
  .grid {{ max-width: 1400px; margin: 0 auto;
           display: grid; gap: 24px;
           grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); }}
  .card {{ background: var(--panel); border-radius: 14px; overflow: hidden;
           border: 1px solid #21262d; transition: transform .12s ease,
           border-color .12s ease; }}
  .card:hover {{ transform: translateY(-2px); border-color: var(--magenta); }}
  .card img {{ display: block; width: 100%; height: auto; }}
  .meta {{ padding: 12px 16px; font-family: ui-monospace, SFMono-Regular,
           Menlo, monospace; font-size: 12px; color: var(--dim);
           display: flex; justify-content: space-between; }}
</style></head><body>
<header>
  <h1>DOOM RL Arena — Model Cards</h1>
  <div class="sub">{n} models · generated {ts}</div>
</header>
<div class="grid">
{tiles}
</div></body></html>
"""

TILE_HTML = (
    '<a class="card" href="{file}" target="_blank">'
    '<img src="{file}" alt="{title}" loading="lazy">'
    '<div class="meta"><span>{title}</span><span>{algo}</span></div></a>'
)


def write_gallery(cards: list[tuple[str, CardData]], out_dir: str) -> None:
    tiles = "\n".join(
        TILE_HTML.format(
            file=os.path.basename(path),
            title=f"{cd.algo.upper()} · {cd.scenario}",
            algo=_short_id(cd.model_id),
        )
        for path, cd in cards
    )
    html = GALLERY_HTML.format(
        n=len(cards),
        ts=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        tiles=tiles,
    )
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(html)


# --- main -------------------------------------------------------------------

def _normalise_efficiency(personalities: dict[str, Personality],
                            scenarios: dict[str, str]) -> dict[str, float]:
    """Per-scenario min-max normalise the raw efficiency. Single-model
    scenarios get 0.5 so they're not visually empty."""
    by_scen: dict[str, list[tuple[str, float]]] = {}
    for mid, p in personalities.items():
        by_scen.setdefault(scenarios[mid], []).append((mid, p.efficiency))
    norm: dict[str, float] = {}
    for items in by_scen.values():
        vals = np.array([v for _, v in items])
        lo, hi = float(vals.min()), float(vals.max())
        for mid, v in items:
            if hi - lo < 1e-9:
                norm[mid] = 0.5
            else:
                norm[mid] = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    return norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=os.path.join(REPO_ROOT, "models/registry.json"))
    ap.add_argument("--out-dir", default="/mnt/orbit_war_disk/doom-checkpoints/cards")
    ap.add_argument("--wandb-entity", default="ignaneswara-srm-institute-of-science-and-technology")
    ap.add_argument("--wandb-project", default="doom-rl")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--only", default=None,
                    help="Comma-separated model ids; render only these (debug).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.registry) as fh:
        registry = json.load(fh)
    entries = registry["models"]
    if args.only:
        wanted = set(args.only.split(","))
        entries = [m for m in entries if m["id"] in wanted]

    import wandb
    api = wandb.Api(timeout=30)

    failures: list[tuple[str, str]] = []
    cards_data: dict[str, CardData] = {}
    paths: dict[str, str] = {}

    for m in entries:
        algo = m.get("algo", "ppo").lower()
        run_name = f"{algo}_{m['scenario']}"
        # Original PPO runs used the bare scenario name in W&B; fall back to
        # that if the {algo}_{scenario} match misses.
        fallbacks = [m["scenario"]] if algo == "ppo" else []
        print(f"[{m['id']}] {run_name} -- fetching curve")
        try:
            curve, wall = fetch_curve(api, args.wandb_entity,
                                       args.wandb_project, run_name,
                                       fallback_names=fallbacks)
        except Exception as e:
            print(f"  W&B fetch failed: {e}")
            curve, wall = Curve(np.array([]), np.array([]), False), None

        print(f"[{m['id']}] {run_name} -- probing personality")
        try:
            personality = probe_personality(
                m["path"], algo, m["scenario"],
                n_episodes=args.episodes, max_steps=args.max_steps,
            )
        except Exception as e:
            print(f"  rollout failed: {e}")
            failures.append((m["id"], f"rollout: {e}"))
            continue

        cards_data[m["id"]] = CardData(
            model_id=m["id"], name=m["name"], algo=algo,
            scenario=m["scenario"], timesteps=int(m["total_timesteps"]),
            mean_reward=float(m["mean_reward"]),
            wall_seconds=wall, curve=curve, personality=personality,
        )
        paths[m["id"]] = os.path.join(args.out_dir, f"{algo}_{m['scenario']}.png")

    eff_norm = _normalise_efficiency(
        {mid: cd.personality for mid, cd in cards_data.items()},
        {mid: cd.scenario for mid, cd in cards_data.items()},
    )

    rendered: list[tuple[str, CardData]] = []
    for mid, cd in cards_data.items():
        path = paths[mid]
        try:
            render_card(cd, eff_norm[mid], path)
            rendered.append((path, cd))
            print(f"  -> {path}")
        except Exception as e:
            print(f"  render failed for {mid}: {e}")
            failures.append((mid, f"render: {e}"))

    write_gallery(rendered, args.out_dir)
    print(f"\nWrote {len(rendered)} cards to {args.out_dir}")
    print(f"Gallery: {os.path.join(args.out_dir, 'index.html')}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for mid, msg in failures:
            print(f"  {mid}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
