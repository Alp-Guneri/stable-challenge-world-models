"""Extra checks on the analysis (same data, split and tracking as data_findings.py).

    .venv/bin/python experiments/data_analysis/checks.py               (run data_findings.py first; ~2 min)

1. split sizes per environment, and how often the example trainer draws each environment
2. holes in the "where the dot goes" map (would distort the 2-px edge-strip numbers)
3. is the dot's centre ever seen inside the Environment 2 wall?
4. action alignment: R^2 of the 1-frame move from ONE action, shifted -3..+3 frames
5. near-wall vs away-from-wall windows: R^2 AND error in pixels for the "actions + recent motion" rule
6. is the scene (everything away from the dot) the same in every clip?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (for corpus_loader.py)
sys.path[:0] = [str(HERE), str(ROOT)]
RESULTS = HERE / "results"

import corpus_loader  # noqa: E402
from data_findings import FUTURE_SPAN, HIST_SPAN, dot_mask, env_of, fit_predict, r2, to_pixel, track  # noqa: E402

LAGS = range(-3, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "corpus"))
    p.add_argument("--stats", default=str(RESULTS))
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    c = corpus_loader.load(a.corpus)
    env = env_of(c.lengths)
    train_eps, val_eps = corpus_loader.episode_split(c, a.val_frac)
    res = {
        "analysis_clips_by_env": {k: int((env[train_eps] == k).sum()) for k in (1, 2)},
        "heldout_clips_by_env": {k: int((env[val_eps] == k).sum()) for k in (1, 2)},
        # example_train.py weights each window by 1/(windows in its clip): every clip is drawn equally often
        "example_trainer_draw_share_by_env": {k: round(float((env == k).mean()), 6) for k in (1, 2)},
    }

    # 6. is the scene the same in every clip? Each first frame vs the per-pixel median of all first frames,
    #    ignoring the dot and 7 px around it (the red dot has a soft glow a few pixels wide)
    for k in (1, 2):
        eps = [e for e in train_eps if env[e] == k]
        first = np.stack([np.asarray(c.frames[c.offsets[e]]) for e in eps])
        near_dot = np.stack([ndimage.binary_dilation(m, iterations=7) for m in dot_mask(first, k)])
        diff = np.abs(first.astype(np.int16) - np.median(first, axis=0)).max(-1)
        diff[near_dot] = 0
        res[f"env{k}_first_frames_max_pixel_difference_away_from_dot"] = int(diff.max())

    heats = {k: np.load(Path(a.stats) / f"heat_env{k}.npy") for k in (1, 2)}
    strips = {}
    for k in (1, 2):
        visited = heats[k] > 0
        holes = ndimage.binary_fill_holes(visited) & ~visited
        strips[k] = visited & ~ndimage.binary_erosion(visited, iterations=2)
        res[f"env{k}_visited_px"] = int(visited.sum())
        res[f"env{k}_holes_in_visited_px"] = int(holes.sum())

    # Environment 2 wall: columns 29-34 everywhere except the door rows 25-34 (data_findings.py, stats.json)
    rows, cols = np.mgrid[:64, :64]
    in_wall = (cols >= 29) & (cols <= 34) & ~((rows >= 25) & (rows <= 34))
    res["env2_frames_with_dot_centre_inside_wall"] = int(heats[2][in_wall].sum())
    res["env2_frames_total_tracked"] = int(heats[2].sum())

    for k in (1, 2):
        eps = np.array([e for e in train_eps if env[e] == k])
        splits = [set(np.random.default_rng(s).choice(eps, size=len(eps) // 5, replace=False).tolist())
                  for s in range(5)]
        tracks = {e: track(np.asarray(c.frames[c.offsets[e]:c.offsets[e + 1]]), k)[0] for e in eps}
        acts = {e: np.asarray(c.actions[c.offsets[e]:c.offsets[e + 1]]) for e in eps}

        # 4. one action at a time, shifted: which single action best predicts the move from frame t to t+1?
        clip, move, lagged = [], [], []
        for e in eps:
            pos = tracks[e]
            for t in range(HIST_SPAN, c.lengths[e] - FUTURE_SPAN):
                d = pos[t + 1] - pos[t]
                if np.isfinite(d).all():
                    clip.append(e)
                    move.append(d)
                    lagged.append(acts[e][t + LAGS.start:t + LAGS.stop])
        clip, move, lagged = np.array(clip), np.array(move), np.array(lagged)
        lag_r2 = {}
        for j, lag in enumerate(LAGS):
            sc = []
            for test in splits:
                te = np.isin(clip, list(test))
                sc.append(r2(move[te], fit_predict(lagged[~te, j], move[~te], lagged[te, j])))
            lag_r2[lag] = round(float(np.mean(sc)), 6)
        res[f"env{k}_1frame_move_r2_from_single_action_at_lag"] = lag_r2

        # 5. near-wall vs away-from-wall, in R^2 and in pixels
        out = {}
        for h in (1, 5, 10):
            clip, recent, pushes, target, touches = [], [], [], [], []
            for e in eps:
                pos = tracks[e]
                for t in range(HIST_SPAN, c.lengths[e] - FUTURE_SPAN):
                    pts = pos[[t - 10, t - 5, t, t + h]]
                    seg = pos[t:t + h + 1]
                    if not (np.isfinite(pts).all() and np.isfinite(seg).all()):
                        continue
                    clip.append(e)
                    recent.append(np.concatenate([pts[2] - pts[1], pts[1] - pts[0]]))
                    pushes.append(acts[e][t:t + h].reshape(-1))
                    target.append(pts[3] - pts[2])
                    ij = to_pixel(seg)
                    touches.append(bool(strips[k][ij[:, 1], ij[:, 0]].any()))
            clip, target, touches = np.array(clip), np.array(target), np.array(touches)
            x = np.concatenate([np.array(recent), np.array(pushes)], 1)
            acc = {g: {"r2": [], "rms_error_px": [], "rms_spread_px": []} for g in ("never_touches", "touches")}
            for test in splits:
                te = np.isin(clip, list(test))
                pred = fit_predict(x[~te], target[~te], x[te])
                y, tt = target[te], touches[te]
                if (~tt).sum() < 500:
                    continue
                for g, m in (("never_touches", ~tt), ("touches", tt)):
                    acc[g]["r2"].append(r2(y[m], pred[m]))
                    acc[g]["rms_error_px"].append(float(np.sqrt(((y[m] - pred[m]) ** 2).sum(1).mean())))
                    acc[g]["rms_spread_px"].append(float(np.sqrt(((y[m] - y[m].mean(0)) ** 2).sum(1).mean())))
            if acc["touches"]["r2"]:
                out[h] = {g: {m: round(float(np.mean(v)), 6) for m, v in d.items()} for g, d in acc.items()}
        res[f"env{k}_both_rule_by_wall_contact"] = out

    Path(a.stats, "checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
