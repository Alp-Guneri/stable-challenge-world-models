"""Where the dot goes in each environment, and how much of its movement a straight-line rule can predict.

    .venv/bin/python experiments/data_analysis/data_findings.py        (from the repo root; ~3 min on CPU)

Reads train.npz exactly as the repo does (corpus_loader: frames, actions, offsets; actions[i] is the
action taken right after frames[i], per CHALLENGE.md). The dataset has no environment labels, so:

  Environment 1 = the 100-frame clips (green dot in a U-shaped corridor)
  Environment 2 = the 50-frame clips  (red dot in two rooms joined by a door in a wall)

Clip counts, and the check that clip length matches each clip's first frame, use all 3,332 clips.
Everything else uses only the training split (the first 90% of clips, corpus_loader.episode_split with
val_frac=0.1), so the dynamics of the 333 clips held out for proxy_eval.py are never looked at.

The dot's position is not stored in the data, so we measure it from the pixels: the centroid of the
dot-coloured pixels in each frame (green in Environment 1, red in Environment 2).
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

HIST_SPAN, FUTURE_SPAN = 10, 25        # the model sees frames t-10, t-5, t and the actions a[t], ..., a[t+24]
HORIZONS = [1, 5, 10, 15, 20, 25]      # frames ahead (5 frames = one model step)


def env_of(lengths):
    return np.where(lengths == 100, 1, 2)


def dot_mask(frames, env):
    """uint8 [T, 64, 64, 3] -> bool [T, 64, 64]: pixels with the dot's colour."""
    f = frames.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    if env == 1:
        return (g > 150) & (r < 150) & (b < 150)   # green dot
    return (r > 180) & (g < 120) & (b < 120)       # red dot


def track(frames, env):
    """Centroid (x, y) in pixels per frame, the number of dot pixels, and the number of separate blobs."""
    m = dot_mask(frames, env)
    n = m.sum((1, 2))
    xs, ys = np.arange(64), np.arange(64)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (m * xs[None, None, :]).sum((1, 2)) / n
        y = (m * ys[None, :, None]).sum((1, 2)) / n
    pos = np.stack([x, y], 1)
    pos[n < 2] = np.nan
    blobs = np.array([ndimage.label(mi)[1] for mi in m])
    return pos, n, blobs


def to_pixel(p):
    """Pixel index of a sub-pixel position, rounding .5 UP. (np.round sends .5 to the nearest EVEN pixel; 13-17% of
    dot centres sit exactly on .5, so np.round would paint a checkerboard into the map.)"""
    return np.floor(np.asarray(p) + 0.5).astype(int).clip(0, 63)


def wall_geometry(first_frame):
    """Environment 2: wall columns and the open (door) rows inside them, from a clip's first frame."""
    black = first_frame.max(-1) < 40
    cols = np.flatnonzero(black.sum(0) > 20)
    if not len(cols):
        return None
    open_rows = np.flatnonzero(~black[:, cols].any(1))
    inner = open_rows[(open_rows > 5) & (open_rows < 58)]        # skip the frame border
    return int(cols.min()), int(cols.max()), (int(inner.min()), int(inner.max())) if len(inner) else None


def r2(y_true, y_pred):
    """Variance-weighted R^2 over the x and y columns (1 = perfect, 0 = no better than the mean)."""
    sse = ((y_true - y_pred) ** 2).sum()
    sst = ((y_true - y_true.mean(0)) ** 2).sum()
    return float(1 - sse / sst)


def fit_predict(xtr, ytr, xte):
    a = np.concatenate([xtr, np.ones((len(xtr), 1))], 1)
    coef, *_ = np.linalg.lstsq(a, ytr, rcond=None)
    return np.concatenate([xte, np.ones((len(xte), 1))], 1) @ coef


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(ROOT / "corpus"))
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    c = corpus_loader.load(a.corpus)
    env = env_of(c.lengths)
    train_eps, _ = corpus_loader.episode_split(c, a.val_frac)
    stats = {"clips_total": len(c), "analysis_clips": len(train_eps)}

    # 1. environments: clip length vs. how the first frame looks (blue mean < 150 = the orange and blue U-corridor)
    looks_env1 = np.array([np.asarray(c.frames[c.offsets[e]])[..., 2].mean() < 150 for e in range(len(c))])
    stats["env_label_agreement"] = float(np.mean(looks_env1 == (env == 1)))

    in_analysis = np.isin(np.arange(len(c)), train_eps)
    for k in (1, 2):
        eps = np.flatnonzero(env == k)                           # all clips: dataset counts only
        ana = np.flatnonzero((env == k) & in_analysis)           # analysis clips: everything else
        acts = [np.asarray(c.actions[c.offsets[e]:c.offsets[e + 1]]) for e in ana]
        act = np.concatenate(acts)
        windows = int(sum(c.lengths[e] - HIST_SPAN - FUTURE_SPAN for e in eps))
        hist, edges = np.histogram(act[:, 0], bins=9)
        autocorr = {}
        for j in (1, 5, 25):                                     # correlation of a[t] with a[t+j], per axis
            a0 = np.concatenate([x[:-j] for x in acts])
            a1 = np.concatenate([x[j:] for x in acts])
            autocorr[j] = [round(float(np.corrcoef(a0[:, d], a1[:, d])[0, 1]), 6) for d in (0, 1)]
        stats[f"env{k}"] = {
            "clips": int(len(eps)), "analysis_clips": int(len(ana)), "heldout_clips": int(len(eps) - len(ana)),
            "frames_per_clip": sorted(set(c.lengths[eps].tolist())),
            "frames": int(c.lengths[eps].sum()), "distinct_windows": windows,
            "action_min": act.min(0).round(3).tolist(), "action_max": act.max(0).round(3).tolist(),
            "action_mean": act.mean(0).round(3).tolist(), "action_std": act.std(0).round(3).tolist(),
            "action_x_hist_share": (hist / hist.sum()).round(3).tolist(),
            "action_x_hist_edges": edges.round(2).tolist(),
            "action_autocorr_by_lag": autocorr,
        }
    total_w = stats["env1"]["distinct_windows"] + stats["env2"]["distinct_windows"]
    for k in (1, 2):
        # share of distinct (clip, t) windows. This is NOT the training mix: example_train.py weights each window
        # by 1 / (windows in its clip), so every clip is drawn equally often (draw share = clip share).
        stats[f"env{k}"]["distinct_window_share"] = round(stats[f"env{k}"]["distinct_windows"] / total_w, 6)
        stats[f"env{k}"]["example_trainer_draw_share"] = round(float(np.mean(env == k)), 6)

    # 2. track the dot in every frame of the analysis clips
    tracks, heat = {}, {1: np.zeros((64, 64)), 2: np.zeros((64, 64))}
    sizes, blobs_all, geo = {1: [], 2: []}, {1: [], 2: []}, []
    for e in train_eps:
        k = int(env[e])
        fr = np.asarray(c.frames[c.offsets[e]:c.offsets[e + 1]])
        pos, n, blobs = track(fr, k)
        tracks[e] = pos
        sizes[k].append(n)
        blobs_all[k].append(blobs)
        ok = np.isfinite(pos).all(1)
        np.add.at(heat[k], (to_pixel(pos[ok, 1]), to_pixel(pos[ok, 0])), 1)
        if k == 2:
            geo.append((e, wall_geometry(fr[0]), pos))
    for k in (1, 2):
        s, bl = np.concatenate(sizes[k]), np.concatenate(blobs_all[k])
        stats[f"env{k}"].update({
            "dot_found_share": round(float(np.mean(s >= 2)), 5),
            "dot_pixels_p1_p50_p99": np.percentile(s[s >= 2], [1, 50, 99]).tolist(),
            "frames_with_more_than_one_blob_share": round(float(np.mean(bl > 1)), 5),
        })
    np.save(out / "heat_env1.npy", heat[1])
    np.save(out / "heat_env2.npy", heat[2])
    # time spent next to a wall or edge: the 2-px strip along the boundary of every position the dot reached
    for k in (1, 2):
        visited = heat[k] > 0
        strip = visited & ~ndimage.binary_erosion(visited, iterations=2)
        stats[f"env{k}"]["time_within_2px_of_edge"] = {
            "area_share": round(float(strip.sum() / visited.sum()), 6),
            "frames_share": round(float(heat[k][strip].sum() / heat[k].sum()), 6)}

    # 3. Environment 2 geometry and door crossings
    walls = [g for _, g, _ in geo if g is not None]
    stats["env2"]["wall_columns_unique"] = sorted({(w[0], w[1]) for w in walls})
    stats["env2"]["door_rows_unique"] = sorted({w[2] for w in walls if w[2]})
    stats["env2"]["geometry_found_in_clips"] = f"{len(walls)}/{len(geo)}"
    # a clip "crosses" if the dot is seen fully left AND fully right of the wall (margin m px beyond its columns)
    cross = {m: 0 for m in (0, 1, 2, 3)}
    for _, g, pos in geo:
        x0, x1 = g[0], g[1]
        x = pos[np.isfinite(pos).all(1), 0]
        for m in cross:
            cross[m] += int((x < x0 - m).any() and (x > x1 + m).any())
    stats["env2"]["clips_crossing_door"] = f"{cross[1]}/{len(geo)}"
    stats["env2"]["clips_crossing_door_share"] = round(cross[1] / len(geo), 6)
    stats["env2"]["crossing_share_by_margin_px"] = {m: round(v / len(geo), 6) for m, v in cross.items()}

    # 4. how predictable is the dot's movement? Straight-line (least-squares) fits from what the model gets:
    #    recent motion    = the two displacements between the input frames t-10 -> t-5 -> t,
    #    upcoming actions = the h actions a[t], ..., a[t+h-1] taken between frame t and frame t+h.
    #    Target = displacement from frame t to frame t+h. Same windows t as the trainer (10 <= t <= length-26).
    #    Each fit is scored on 20% of clips it never saw; repeated over 5 random clip splits.
    #    (Near-wall vs away-from-wall comparisons live in checks.py, with errors in pixels as well as R^2.)
    for k in (1, 2):
        visited = heat[k] > 0
        strip = visited & ~ndimage.binary_erosion(visited, iterations=2)
        eps = np.array([e for e in train_eps if env[e] == k])
        splits = [set(np.random.default_rng(s).choice(eps, size=len(eps) // 5, replace=False).tolist())
                  for s in range(5)]
        res = {}
        for h in HORIZONS:
            clip, recent, upcoming, target, touches = [], [], [], [], []
            for e in eps:
                pos = tracks[e]
                act = np.asarray(c.actions[c.offsets[e]:c.offsets[e + 1]])
                for t in range(HIST_SPAN, c.lengths[e] - FUTURE_SPAN):
                    pts = pos[[t - 10, t - 5, t, t + h]]
                    seg = pos[t:t + h + 1]
                    if not (np.isfinite(pts).all() and np.isfinite(seg).all()):
                        continue
                    clip.append(e)
                    recent.append(np.concatenate([pts[2] - pts[1], pts[1] - pts[0]]))
                    upcoming.append(act[t:t + h].reshape(-1))
                    target.append(pts[3] - pts[2])
                    ij = to_pixel(seg)
                    touches.append(bool(strip[ij[:, 1], ij[:, 0]].any()))
            clip, recent, upcoming = np.array(clip), np.array(recent), np.array(upcoming)
            target, touches = np.array(target), np.array(touches)
            feats = {"actions_only": upcoming, "recent_motion_only": recent,
                     "both": np.concatenate([recent, upcoming], 1)}
            scores = {name: [] for name in feats}
            for test in splits:
                te = np.isin(clip, list(test))
                for name, x in feats.items():
                    scores[name].append(r2(target[te], fit_predict(x[~te], target[~te], x[te])))
            out_h = {"windows": int(len(target)),
                     "median_displacement_px": round(float(np.median(np.linalg.norm(target, axis=1))), 4),
                     "windows_touching_wall_or_edge_share": round(float(touches.mean()), 6)}
            for name, v in scores.items():
                out_h[name] = round(float(np.mean(v)), 6)
                out_h[name + "_range_over_splits"] = [round(float(min(v)), 6), round(float(max(v)), 6)]
            res[str(h)] = out_h
        stats[f"env{k}"]["predictability_heldout_r2"] = res

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
