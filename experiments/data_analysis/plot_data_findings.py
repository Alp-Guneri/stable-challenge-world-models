"""Figures from data_findings.py's output: one per environment, plus a check of the dot tracking.

    .venv/bin/python experiments/data_analysis/plot_data_findings.py   (run data_findings.py first)

Writes env1.png, env2.png (same layout and scales) and tracking_check.png (the detected dot centre on random frames).
Every number drawn comes from stats.json; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (for corpus_loader.py)
sys.path[:0] = [str(HERE), str(ROOT)]
RESULTS = HERE / "results"

import corpus_loader  # noqa: E402
from data_findings import env_of, track  # noqa: E402

INK, INK2, GRID, SURFACE = "#1f1f1d", "#5f5e5a", "#e4e3df", "#ffffff"
BOTH, ACTIONS, RECENT = "#2a78d6", "#eb6834", "#1baf7a"   # one colour per information source, same in both figures
HEAT = LinearSegmentedColormap.from_list("heat", [(1, 1, 1, 0), "#9085e9", "#3b2db0"])
TITLES = {1: "Environment 1: green dot in a U-shaped corridor",
          2: "Environment 2: red dot in two rooms joined by a door"}


def num(v, fmt=".1f"):
    """Number with a real minus sign, and no '-0.0'."""
    s = format(0.0 if abs(v) < 0.5 * 10 ** -int(fmt[1]) else v, fmt)
    return s.replace("-", "−")


def spread(values, gaps):
    """Label heights, in value order, keeping at least gaps[i] between label i and the one below it."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        placed[b] = max(placed[b], placed[a] + max(gaps[a], gaps[b]))
    return placed


def background(c, eps):
    """The room without the dot: per-pixel median over many clips' first frames, faded toward white."""
    firsts = np.stack([np.asarray(c.frames[c.offsets[e]]) for e in eps[:200]])
    gray = np.median(firsts, axis=0).mean(-1) / 255.0
    return 0.70 + 0.30 * gray


def env_figure(k, c, stats, heat, eps, out):
    s = stats[f"env{k}"]
    pr = s["predictability_heldout_r2"]
    hs = [int(h) for h in pr]
    plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(13.33, 5.5), dpi=200, facecolor=SURFACE)
    fig.text(0.02, 0.945, TITLES[k], fontsize=17, fontweight="bold", color=INK)
    lag1 = float(np.mean(s["action_autocorr_by_lag"]["1"]))
    fig.text(0.02, 0.898, f"{s['clips']:,} clips × {s['frames_per_clip'][0]} frames · actions "
             f"{num(min(s['action_min']))} to {num(max(s['action_max']))} on each axis · "
             f"correlation between consecutive actions: {num(lag1, '.2f')}", fontsize=12, color=INK2)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.75], wspace=0.27, left=0.02, right=0.80, top=0.74, bottom=0.21)

    # left: where the dot spends its time
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(background(c, eps), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    h = heat / heat.sum()
    cap = np.percentile(h[h > 0], 95)      # linear scale; the busiest 5% of pixels share the darkest shade
    im = ax.imshow(np.ma.masked_where(h <= 0, np.clip(h / cap, 0, 1)), cmap=HEAT, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.set_title("Where the dot spends its time", fontsize=13, loc="left", color=INK, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0.05, 0.95])
    cb.ax.set_yticklabels(["fewer\nframes", "more\nframes"], fontsize=9, color=INK2)
    cb.outline.set_visible(False)
    edge = s["time_within_2px_of_edge"]
    ax.text(0, -0.06, f"{edge['frames_share']:.0%} of frames: dot within 2 px of a wall or edge\n"
            f"(that strip is {edge['area_share']:.0%} of the area the dot reaches)",
            transform=ax.transAxes, fontsize=10.5, color=INK2, va="top")
    if k == 2:
        ax.annotate("door", xy=(31.5, 29.5), xytext=(44, 20), fontsize=10.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=1))

    # right: how predictable the movement is
    ax = fig.add_subplot(gs[0, 1])
    series = [("both", "Upcoming actions and\nrecent motion", BOTH, "-", 3.0),
              ("actions_only", "Upcoming actions only", ACTIONS, "-", 2.0),
              ("recent_motion_only", "Recent motion only\n(from the 3 input frames)", RECENT, "--", 2.0)]
    curves = [[pr[str(h)][key] for h in hs] for key, *_ in series]
    lines = [label.count("\n") + 1 for _, label, *_ in series]
    end_y = spread([ys[-1] for ys in curves], [0.055 * n for n in lines])
    start_y = spread([ys[0] for ys in curves], [0.05] * len(series))
    for (key, label, col, ls, lw), ys, ey, sy in zip(series, curves, end_y, start_y):
        ax.plot(hs, ys, color=col, ls=ls, lw=lw, marker="o", ms=5, zorder=3)
        ax.plot([25.35, 26.05], [ys[-1], ey], color=col, lw=1.2, clip_on=False)     # leader to the label
        ax.text(26.25, ey, f"{label}: {ys[-1]:.2f}", color=INK, fontsize=10.5, va="center")
        ax.text(hs[0] - 0.6, sy, f"{ys[0]:.2f}", color=INK2, fontsize=10, va="center", ha="right")
    ax.set_xlim(-1.5, 26)
    ax.set_ylim(0, 1)
    ax.set_xticks(hs)
    ax.set_xticklabels(["1", "5\n(1 model step)", "10", "15", "20", "25\n(5 model steps)"], fontsize=10.5)
    ax.set_xlabel("Frames ahead", fontsize=11)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("R² on unseen clips", fontsize=11)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("How well a straight-line rule predicts the dot's movement\n"
                 "(1 = perfect, 0 = no better than always guessing the average movement)",
                 fontsize=13, loc="left", color=INK, pad=6)
    fig.text(0.02, 0.052, "Dot position: centre of the dot-coloured pixels in each frame. Analysis uses "
             f"{s['analysis_clips']:,} of the {s['clips']:,} clips ({s['heldout_clips']} held out for "
             "evaluating our models, not used).", fontsize=9.5, color=INK2)
    fig.text(0.02, 0.018, f"R²: straight-line rule fitted on 80% of those {s['analysis_clips']:,} clips and "
             "scored on the other 20%; mean of 5 random splits.", fontsize=9.5, color=INK2)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def tracking_figure(c, env, eps_by_env, out, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 6, figsize=(13.33, 5.6), dpi=200, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.32})
    for row, k in enumerate((1, 2)):
        for col, e in enumerate(rng.choice(eps_by_env[k], 6, replace=False)):
            fr = np.asarray(c.frames[c.offsets[e]:c.offsets[e + 1]])
            t = int(rng.integers(len(fr)))
            pos, _, _ = track(fr[t:t + 1], k)
            ax = axes[row, col]
            ax.imshow(fr[t], interpolation="nearest")
            ax.plot(pos[0, 0], pos[0, 1], marker="+", ms=16, mew=1.6, color="black")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"clip {e}, frame {t + 1}", fontsize=9, color=INK2)
        axes[row, 0].set_ylabel(f"Environment {k}", fontsize=11, color=INK)
    fig.suptitle("How we find the dot: centre of the dot-coloured pixels (black +), random frames",
                 fontsize=13, x=0.02, ha="left", color=INK)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.02, wspace=0.08, hspace=0.32)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "corpus"))
    p.add_argument("--stats", default=str(RESULTS))
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    c = corpus_loader.load(a.corpus)
    env = env_of(c.lengths)
    train, _ = corpus_loader.episode_split(c, a.val_frac)
    stats = json.loads((Path(a.stats) / "stats.json").read_text())
    eps_by_env = {k: [e for e in train if env[e] == k] for k in (1, 2)}
    for k in (1, 2):
        heat = np.load(Path(a.stats) / f"heat_env{k}.npy")
        env_figure(k, c, stats, heat, eps_by_env[k], Path(a.out) / f"env{k}.png")
    tracking_figure(c, env, eps_by_env, Path(a.out) / "tracking_check.png")


if __name__ == "__main__":
    main()
