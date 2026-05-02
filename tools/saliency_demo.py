"""Generate a saliency-overlay video for a trained checkpoint.

Rolls one greedy episode in a scenario, computes perturbation saliency every
--every frames, and writes:

  <out>/raw.mp4         raw RGB game frames (the env's 640x480 buffer, upsampled)
  <out>/policy.mp4      raw + policy-saliency overlay
  <out>/value.mp4       raw + value-saliency overlay
  <out>/strip.png       6-frame strip of the highest-surprise frames (for sharing)
  <out>/manifest.json   metadata (rewards, KL spikes, sanity-check result)

Per Velez's review: contour-line fallback toggle, perceptually-uniform colormap.
Per Iyer's review: policy and value saliency are kept separate, never blended.
                   sanity_check() is run once per demo and embedded in manifest.

Usage
=====
    python tools/saliency_demo.py \\
        --checkpoint /mnt/orbit_war_disk/doom-checkpoints/basic/basic_final.zip \\
        --scenario basic \\
        --out /mnt/orbit_war_disk/doom-checkpoints/saliency-demos/basic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import imageio.v2 as imageio
from matplotlib import cm

from stable_baselines3 import PPO

from doom.environment import VizDoomGym
from doom.saliency import PerturbationSaliency, SaliencyConfig, sanity_check


# ─── overlay rendering ──────────────────────────────────────────────────────

def _to_heatmap(sal: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Saliency map (H, W) in [0, 1] -> RGBA (H, W, 4) uint8 via colormap."""
    cmap = cm.get_cmap(cmap_name) if hasattr(cm, "get_cmap") else __import__("matplotlib").colormaps.get_cmap(cmap_name)
    rgba = (cmap(sal) * 255).astype(np.uint8)
    return rgba


def _overlay(rgb: np.ndarray, sal: np.ndarray, opacity: float = 0.55,
             cmap_name: str = "magma") -> np.ndarray:
    """Resize saliency map to rgb's shape, colourise, alpha-blend.

    rgb: (H, W, 3) uint8 game frame
    sal: (h, w) float in [0, 1]
    """
    import cv2
    H, W = rgb.shape[:2]
    sal_resized = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    sal_resized = np.clip(sal_resized, 0, 1)
    heat = _to_heatmap(sal_resized, cmap_name=cmap_name)[..., :3]  # drop alpha
    blended = (rgb.astype(np.float32) * (1 - opacity) +
               heat.astype(np.float32) * opacity).clip(0, 255).astype(np.uint8)
    return blended


def _overlay_contours(rgb: np.ndarray, sal: np.ndarray,
                      n_levels: int = 6) -> np.ndarray:
    """Color-blind-safe alternative: draw contour lines instead of a heatmap.

    Per Velez's review — the diverging-palette risk for deuteranopes is real.
    """
    import cv2
    H, W = rgb.shape[:2]
    sal_resized = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    out = rgb.copy()
    for level in np.linspace(0.4, 0.95, n_levels):
        mask = (sal_resized >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        # Brighter line for higher levels
        intensity = int(80 + 175 * (level - 0.4) / 0.55)
        cv2.drawContours(out, contours, -1, (intensity, intensity, 255), 1)
    return out


# ─── episode rollout ────────────────────────────────────────────────────────

def rollout_with_saliency(model, env, sal: PerturbationSaliency,
                           every: int, max_steps: int, deterministic: bool):
    """Roll one episode. Return list of dicts: {raw_rgb, obs, action, reward,
    pol_map, val_map, kl_action, value} where saliency is None except every Kth
    step."""
    obs, _ = env.reset()
    frames = []
    for step in range(max_steps):
        # SB3 predict
        action, _ = model.predict(obs, deterministic=deterministic)

        # Get the env's RGB buffer for nice rendering
        raw_rgb = env.get_raw_frame()
        if raw_rgb is None:
            raw_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # Compute saliency every K steps
        pol = val = None
        if step % every == 0:
            pol, val = sal.both(obs)

        frames.append({
            "step": step,
            "raw_rgb": raw_rgb,
            "obs": obs,
            "action": int(action),
            "pol": pol,
            "val": val,
        })

        obs, reward, terminated, truncated, info = env.step(action)
        frames[-1]["reward"] = float(reward)
        if terminated or truncated:
            break
    return frames


def fill_saliency_inbetween(frames):
    """For frames where saliency wasn't computed, copy from the nearest
    earlier frame (so playback looks continuous instead of flickering)."""
    last_pol = None
    last_val = None
    for f in frames:
        if f["pol"] is not None:
            last_pol = f["pol"]
        else:
            f["pol"] = last_pol
        if f["val"] is not None:
            last_val = f["val"]
        else:
            f["val"] = last_val
    return frames


# ─── strip selection ────────────────────────────────────────────────────────

def pick_surprise_strip(frames, k: int = 6):
    """Pick K frames with highest 'surprise' = max(policy_saliency).

    Per Velez's review — the highest-surprise window is what we'd auto-cut to a
    GIF for sharing. This is the static version of that.
    """
    keyed = [(i, float(f["pol"].max()) if f["pol"] is not None else 0.0)
             for i, f in enumerate(frames)]
    keyed.sort(key=lambda x: -x[1])
    chosen = sorted([i for i, _ in keyed[:k]])
    return [frames[i] for i in chosen]


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4,
                    help="compute saliency every N env steps")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--stride", type=int, default=8,
                    help="saliency perturbation stride in pixels")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--opacity", type=float, default=0.55)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--no-sanity", action="store_true",
                    help="skip the Adebayo randomisation test (faster)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"loading {args.checkpoint}")
    model = PPO.load(args.checkpoint, device="cuda")

    env = VizDoomGym(args.scenario, render=False)
    sal = PerturbationSaliency(model, SaliencyConfig(stride=args.stride))

    print(f"rolling episode (every={args.every}, max_steps={args.max-steps if False else args.max_steps})")
    t0 = time.time()
    frames = rollout_with_saliency(
        model, env, sal,
        every=args.every,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
    )
    rollout_secs = time.time() - t0
    fill_saliency_inbetween(frames)
    print(f"  episode: {len(frames)} steps in {rollout_secs:.1f}s")

    # Sanity check on the first frame's obs (cheap if --no-sanity)
    sanity = None
    if not args.no_sanity and frames:
        print("running Adebayo randomisation sanity check on first frame...")
        s = sanity_check(model, frames[0]["obs"])
        sanity = {
            "policy_correlation_to_random": s["policy_correlation_to_random"],
            "value_correlation_to_random": s["value_correlation_to_random"],
            "policy_passes_sanity": bool(s["policy_passes_sanity"]),
            "value_passes_sanity": bool(s["value_passes_sanity"]),
        }
        print(f"  policy corr={sanity['policy_correlation_to_random']:+.3f} "
              f"passes={sanity['policy_passes_sanity']}")
        print(f"  value  corr={sanity['value_correlation_to_random']:+.3f} "
              f"passes={sanity['value_passes_sanity']}")

    # Render videos
    print("rendering videos...")
    raw_writer = imageio.get_writer(os.path.join(args.out, "raw.mp4"),
                                     fps=args.fps, codec="libx264", quality=8)
    pol_writer = imageio.get_writer(os.path.join(args.out, "policy.mp4"),
                                     fps=args.fps, codec="libx264", quality=8)
    val_writer = imageio.get_writer(os.path.join(args.out, "value.mp4"),
                                     fps=args.fps, codec="libx264", quality=8)
    contour_writer = imageio.get_writer(os.path.join(args.out, "policy_contour.mp4"),
                                         fps=args.fps, codec="libx264", quality=8)

    for f in frames:
        rgb = f["raw_rgb"]
        raw_writer.append_data(rgb)
        if f["pol"] is not None:
            pol_writer.append_data(_overlay(rgb, f["pol"], args.opacity, "magma"))
            contour_writer.append_data(_overlay_contours(rgb, f["pol"]))
        else:
            pol_writer.append_data(rgb)
            contour_writer.append_data(rgb)
        if f["val"] is not None:
            val_writer.append_data(_overlay(rgb, f["val"], args.opacity, "viridis"))
        else:
            val_writer.append_data(rgb)

    raw_writer.close()
    pol_writer.close()
    val_writer.close()
    contour_writer.close()

    # 6-frame strip of highest-surprise frames
    print("rendering surprise strip...")
    strip_frames = pick_surprise_strip(frames, k=6)
    strip = []
    for f in strip_frames:
        if f["pol"] is None:
            continue
        ov = _overlay(f["raw_rgb"], f["pol"], args.opacity, "magma")
        strip.append(ov)
    if strip:
        # Pad to common width
        max_h = max(im.shape[0] for im in strip)
        max_w = max(im.shape[1] for im in strip)
        padded = []
        for im in strip:
            pad_h = max_h - im.shape[0]
            pad_w = max_w - im.shape[1]
            if pad_h or pad_w:
                im = np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)),
                             mode="constant")
            padded.append(im)
        strip_img = np.concatenate(padded, axis=1)
        imageio.imwrite(os.path.join(args.out, "strip.png"), strip_img)

    # Manifest
    rewards = [f.get("reward", 0.0) for f in frames]
    manifest = {
        "checkpoint": args.checkpoint,
        "scenario": args.scenario,
        "n_steps": len(frames),
        "total_reward": float(sum(rewards)),
        "saliency_stride": args.stride,
        "saliency_every": args.every,
        "fps": args.fps,
        "rollout_seconds": rollout_secs,
        "sanity": sanity,
        "files": {
            "raw": "raw.mp4",
            "policy": "policy.mp4",
            "value": "value.mp4",
            "policy_contour": "policy_contour.mp4",
            "strip": "strip.png",
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {args.out}/")
    print(f"  policy.mp4 ({len(frames)} frames @ {args.fps} fps)")
    print(f"  value.mp4")
    print(f"  policy_contour.mp4   (color-blind safe alternative)")
    print(f"  strip.png            (6 highest-surprise frames)")
    print(f"  manifest.json        total_reward={sum(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    main()
