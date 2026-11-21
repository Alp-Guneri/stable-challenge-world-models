"""Evaluate rollout and representation quality on a split of the public corpus.

    uv run proxy_eval.py model.onnx --corpus corpus --floor floor.onnx --reference ref_holdout.onnx

Reserve the same episodes during training to obtain a held-out score. The challenge leaderboard
uses private data and tasks.

For every held-out window it runs the model twice, exactly as the evaluator would:

    z[:, 1:]  the predictions, from the frames at t-10, t-5, t and the actions from t on
    z[:, 0]   the embedding of each real future observation, from its own three frames

and then reports, per horizon step k,

    top1    how often a prediction's nearest neighbour (cosine) among all candidates is its own
            future, counting only candidates from other episodes
    nmse    squared distance to its own future over the squared distance to an unrelated future
    pixel   ridge probe R^2 for the 16x16 downsampled pixel change from the starting frame
    pos     ridge probe R^2 for the center of motion over the most recent model step

Each ridge probe fits and scores on disjoint episodes, and the encoder variants probe real future
embeddings. With ``--floor`` and ``--reference``, the blend averages the rescaled retrieval and
motion scores. The supplied anchors define 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

import corpus_loader
from evaluate import ACTION_DIM, HIST, HORIZON, SKIP, load_model, run_batches

SPAN_BACK, SPAN_FWD = (HIST - 1) * SKIP, HORIZON * SKIP


def windows(root, val_frac, per_episode, seed):
    """Windows from the held-out episodes: frames uint8 [n, HIST + HORIZON, 64, 64, 3], actions
    float32 [n, HORIZON, 2 * SKIP], and the episode each window came from."""
    c = corpus_loader.load(root)
    _, validation = corpus_loader.episode_split(c, val_frac)
    if not validation.size or per_episode < 1:
        raise ValueError("evaluation requires held-out episodes and at least one window per episode")
    rng = np.random.default_rng(seed)
    frames, actions, episode = [], [], []
    for e in validation:
        choices = np.arange(SPAN_BACK, c.lengths[e] - SPAN_FWD)
        for t in rng.choice(choices, size=min(per_episode, len(choices)), replace=False):
            s = c.offsets[e]
            rows = s + t + np.arange(-SPAN_BACK, SPAN_FWD + 1, SKIP)
            frames.append(np.asarray(c.frames[rows]))
            actions.append(np.asarray(c.actions[s + t : s + t + SPAN_FWD]).reshape(HORIZON, ACTION_DIM * SKIP))
            episode.append(e)
    return np.stack(frames), np.stack(actions).astype(np.float32), np.asarray(episode)


def normalize(embeddings):
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)


def retrieval_hits(u, v, episode):
    """Find the closest normalized target, excluding other windows from the query episode."""
    sim = u @ v.T
    same_episode = episode[:, None] == episode[None, :]
    np.fill_diagonal(same_episode, False)
    sim[same_episode] = -np.inf  # a neighbouring window of the same episode is not a wrong answer
    return sim.argmax(1) == np.arange(len(sim))


def metrics(pred, target, episode):
    """pred, target: [n, D] -> (top-1 retrieval accuracy, normalized squared error)."""
    top1 = float(retrieval_hits(normalize(pred), normalize(target), episode).mean())

    own = np.square(pred - target).sum(1)
    shuffled = np.square(pred - target[np.roll(np.arange(len(target)), 1)]).sum(1)
    return top1, float(own.mean() / max(shuffled.mean(), 1e-12))


def downsample(frames):
    """uint8 [n, 64, 64, 3] -> float [n, 16*16*3] in [0, 1]."""
    n = len(frames)
    x = frames.astype(np.float32).reshape(n, 16, 4, 16, 4, 3).mean(axis=(2, 4))
    return x.reshape(n, -1) / 255.0


def motion_xy(prev, cur):
    """Centroid, in [0, 1], of the pixels that moved between two uint8 frames. Rows with almost
    no motion are NaN and dropped by the probe."""
    d = np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean(-1)
    thr = np.quantile(d, 0.98, axis=(1, 2), keepdims=True)
    w = np.clip(d - thr, 0, None)
    mass = w.sum(axis=(1, 2))
    w = w / np.maximum(mass[:, None, None], 1e-6)
    xs = np.linspace(0, 1, d.shape[-1])
    ys = np.linspace(0, 1, d.shape[-2])
    xy = np.stack([(w * xs[None, None, :]).sum((1, 2)), (w * ys[None, :, None]).sum((1, 2))], 1)
    xy[mass < 1] = np.nan
    return xy


def ridge_r2(x, y, episode, alpha):
    """R^2 of a ridge map x -> y, fit and scored on disjoint episodes."""
    ok = np.isfinite(y).all(1)
    x, y, episode = x[ok], y[ok], episode[ok]
    episodes = np.unique(episode)
    rng = np.random.default_rng(0)
    rng.shuffle(episodes)
    cut = max(1, int(0.7 * len(episodes)))
    fit = np.isin(episode, episodes[:cut])
    test = np.isin(episode, episodes[cut:])
    if fit.sum() < 20 or test.sum() < 10:
        return float("nan")
    mu, sd = x[fit].mean(0), x[fit].std(0)
    sd = np.maximum(sd, 1e-6)

    def norm(z):
        return (z - mu) / sd

    model = Ridge(alpha=alpha)
    model.fit(norm(x[fit]), y[fit])
    return float(model.score(norm(x[test]), y[test]))


def evaluate_model(path, frames, actions, episode, verbose=True):
    sess = load_model(str(path))
    z = run_batches(sess, frames[:, :HIST], actions)  # [n, HORIZON + 1, D]
    rows = []
    for k in range(1, HORIZON + 1):
        # how the evaluator embeds the real observation at t + SKIP*k: its own HIST frames
        target = run_batches(sess, frames[:, k : k + HIST], actions)[:, 0]
        top1, nmse = metrics(z[:, k], target, episode)
        # the same retrieval from the current embedding, dynamics ignored: how much of the score
        # comes from the encoder telling episodes apart rather than from the predictor
        still, _ = metrics(z[:, 0], target, episode)
        idx = HIST - 1 + k
        # Absolute pixels barely change, so R^2 on them saturates for any embedding. Score the
        # change from the current frame, which is the part a world model has to get right.
        delta = downsample(frames[:, idx]) - downsample(frames[:, HIST - 1])
        pos = motion_xy(frames[:, idx - 1], frames[:, idx])
        row = {"k": k, "top1": top1, "nmse": nmse, "encoder_only": still,
               "pixel_r2": ridge_r2(z[:, k], delta, episode, 10.0),
               "pixel_r2_enc": ridge_r2(target, delta, episode, 10.0),
               "pos_r2": ridge_r2(z[:, k], pos, episode, 1.0),
               "pos_r2_enc": ridge_r2(target, pos, episode, 1.0)}
        rows.append(row)
        if verbose:
            print(f"  k={k} (t+{k * SKIP:2d} steps)  top1 {top1:.3f}  nmse {nmse:.3f}"
                  f"  pixel {row['pixel_r2']:.3f} (enc {row['pixel_r2_enc']:.3f})"
                  f"  pos {row['pos_r2']:.3f} (enc {row['pos_r2_enc']:.3f})", file=sys.stderr)
    mean = {key: float(np.mean([r[key] for r in rows]))
            for key in ("top1", "nmse", "encoder_only", "pixel_r2", "pixel_r2_enc", "pos_r2", "pos_r2_enc")}
    mean.update(steps=rows, dim=int(z.shape[2]))
    return mean


def json_results(value):
    """Represent unavailable probe scores as JSON null."""
    if isinstance(value, dict):
        return {key: json_results(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_results(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--corpus", type=Path, default=Path("corpus"))
    p.add_argument("--val-frac", type=float, default=0.1, help="must match training's --val-frac")
    p.add_argument("--per-episode", type=int, default=4, help="windows sampled per held-out episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--floor", type=Path, default=None, help="the model that defines a proxy score of 0")
    p.add_argument("--reference", type=Path, default=None, help="the model that defines a proxy score of 1")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    try:
        frames, actions, episode = windows(args.corpus, args.val_frac, args.per_episode, args.seed)
    except ValueError as error:
        p.error(str(error))
    print(f"{len(frames)} windows from {len(set(episode.tolist()))} held-out episodes", file=sys.stderr)

    paths = list(dict.fromkeys(args.models + [q for q in (args.floor, args.reference) if q]))
    results = {}
    for path in paths:
        print(f"\n{path}", file=sys.stderr)
        results[str(path)] = evaluate_model(path, frames, actions, episode)

    if args.floor and args.reference:
        floor_r, ref_r = results[str(args.floor)], results[str(args.reference)]
        for r in results.values():
            scaled = {}
            for key in ("top1", "pixel_r2", "pos_r2"):
                lo, hi = floor_r[key], ref_r[key]
                scaled[key] = (r[key] - lo) / (hi - lo) if hi > lo else float("nan")
            r["proxy_score"] = scaled["top1"]
            r["scaled"] = scaled
            components = [scaled[key] for key in ("top1", "pos_r2") if math.isfinite(scaled[key])]
            r["blend"] = float(np.mean(components)) if components else float("nan")

    header = "{:<28} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8}".format(
        "model", "top1", "pixel", "pos", "enc-px", "top1~", "blend")
    print("\n" + header, file=sys.stderr)
    for path, r in results.items():
        top = f"{r['proxy_score']:.2f}" if "proxy_score" in r else "-"
        blend = f"{r['blend']:.2f}" if "blend" in r else "-"
        print(f"{Path(path).name:<28} {r['top1']:>7.3f} {r['pixel_r2']:>7.3f} {r['pos_r2']:>7.3f} "
              f"{r['pixel_r2_enc']:>7.3f} {top:>8} {blend:>8}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(json_results(results), indent=2, allow_nan=False) + "\n")
    summary = {k: {"top1": v["top1"], "pixel_r2": v["pixel_r2"], "pos_r2": v["pos_r2"],
                   "blend": v.get("blend")} for k, v in results.items()}
    print(json.dumps(json_results(summary), allow_nan=False))


if __name__ == "__main__":
    main()
