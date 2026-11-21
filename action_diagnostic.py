"""Compare held-out future retrieval and ranking of the recorded action sequence."""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate
from proxy_eval import normalize, retrieval_hits, windows


def score_model(path, frames, actions, episodes, shifts):
    session = evaluate.load_model(str(path))
    start = frames[:, :evaluate.HIST]
    prediction = normalize(evaluate.run_batches(session, start, actions))
    alternatives = [
        normalize(evaluate.run_batches(session, start, np.roll(actions, shift, axis=0)))
        for shift in shifts
    ]
    # Keep the same five action vectors in each block and change only their order.
    reordered = actions.reshape(len(actions), evaluate.HORIZON, evaluate.SKIP, evaluate.ACTION_DIM)
    reordered = reordered[:, :, [2, 0, 4, 1, 3]].reshape(actions.shape)
    reordered_pred = normalize(evaluate.run_batches(session, start, reordered))
    rows = {}
    for step in (4, 5):
        final = frames[:, step:step + evaluate.HIST]
        target = normalize(evaluate.run_batches(session, final, actions)[:, 0])
        retrieved = retrieval_hits(prediction[:, step], target, episodes)
        true_score = np.sum(prediction[:, step] * target, axis=-1)
        alternative_scores = np.stack([np.sum(z[:, step] * target, axis=-1) for z in alternatives])
        action_rank = 1 + np.sum(alternative_scores >= true_score, axis=0)
        reordered_score = np.sum(reordered_pred[:, step] * target, axis=-1)
        rows[step] = (retrieved, action_rank, true_score > reordered_score)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    frames, actions, episodes = windows(args.corpus, val_frac=0.1, per_episode=4, seed=0)
    episode_count = len(np.unique(episodes))
    if episode_count < 8 or len(episodes) != 4 * episode_count:
        parser.error("action diagnosis needs at least eight episodes with four windows each")
    rng = np.random.default_rng(7)
    shifts = 4 * rng.choice(np.arange(1, episode_count), size=7, replace=False)
    bootstrap = rng.integers(0, episode_count, size=(4000, episode_count))
    results = {}
    reference = None
    for path in args.models:
        rows = score_model(path, frames, actions, episodes, shifts)
        if reference is None:
            reference = rows
        result = {}
        for step, (retrieved, action_rank, beats_reordered) in rows.items():
            difference = retrieved.astype(float) - reference[step][0]
            by_episode = difference.reshape(-1, 4).mean(axis=1)
            interval = np.quantile(by_episode[bootstrap].mean(axis=1), [0.025, 0.975])
            result[str(step * evaluate.SKIP)] = {
                "retrieval_top1": float(retrieved.mean()),
                "retrieval_delta_interval_95": interval.tolist(),
                "action_top1_of_8": float((action_rank == 1).mean()),
                "action_mrr_of_8": float((1.0 / action_rank).mean()),
                "recorded_order_beats_reordered": float(beats_reordered.mean()),
            }
        results[str(path)] = result
        print(path.name, json.dumps(result), flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
