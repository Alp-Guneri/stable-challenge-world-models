"""Plots and tables for the ablation.

    .venv/bin/python experiments/ablations/report.py --name main             # every combination + comparison
    .venv/bin/python experiments/ablations/report.py --name main --smooth 1  # no smoothing

Per combination (written by train.py when its seeds finish, or here):
    <combo>/plots/training.png   every logged training metric: mean over seeds, band = min..max over seeds
    <combo>/summary.json         per-seed and aggregated evaluation results
Across combinations (results/<name>/report/):
    training_comparison.png      key training metrics, one panel per loss, one line per encoder-predictor pair
    eval_comparison.png          held-out metrics per combination: mean (bar) and each seed (dots)
    top1_by_horizon.png          held-out retrieval at 5..25 steps ahead
    per_run.csv, summary.csv, summary.md

Curves are smoothed with a trailing rolling mean (default: 1/200 of the run) before the seeds are combined.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Encoder-predictor pairs in a fixed order, each with a fixed categorical color (reference palette, slots 1-5).
ARCHS = ["cnn-mlp", "cnn-ar", "vit-mlp", "vit-ar", "vit-mlp1"]
COLORS = dict(zip(ARCHS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
LOSSES = ["ref", "lewm"]
LOSS_TITLE = {"ref": "Reference loss (MSE + VCReg + InfoNCE)", "lewm": "LeWM loss (MSE + SIGReg)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "font.size": 8.5, "axes.titlesize": 9, "axes.titlelocation": "left", "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round", "legend.frameon": False,
})

# Training panels, in order: (column in metrics.csv, title). Columns not listed here are appended at the end.
PANELS = [
    ("train/loss", "Loss being trained"),
    ("train/ref/total", "Reference loss, total"),
    ("train/lewm/total", "LeWM loss, total"),
    ("train/mse", "Prediction MSE, all steps"),
    *[(f"train/mse_k{k}", f"Prediction MSE, {5 * k} steps ahead") for k in range(1, 6)],
    ("train/ref/infonce", "InfoNCE"),
    ("train/ref/vcreg", "VCReg"),
    ("train/ref/vcreg_t", "VCReg within clip"),
    ("train/lewm/sigreg", "SIGReg"),
    ("train/batch_top1", "In-batch retrieval top-1"),
    ("train/cos_pred_target", "Cosine(prediction, target)"),
    ("train/emb_std", "Embedding std (mean over dims)"),
    ("train/emb_eff_rank", "Embedding effective rank"),
    ("train/pred_std", "Prediction std (mean over dims)"),
    ("train/grad_norm_encoder", "Gradient norm, encoder"),
    ("train/grad_norm_predictor", "Gradient norm, predictor"),
    ("hparams/lr_encoder", "Learning rate, encoder"),
    ("hparams/lr_predictor", "Learning rate, predictor"),
    ("train/step_seconds", "Seconds per step"),
]
KEY_PANELS = ["train/mse", "train/batch_top1", "train/ref/infonce", "train/lewm/sigreg", "train/emb_eff_rank",
              "train/grad_norm_encoder"]
EVAL_METRICS = [("top1", "Held-out retrieval top-1"), ("pos_r2", "Motion R²"), ("blend", "Blend (local)"),
                ("action_top1_25", "Recorded actions ranked 1st of 8 (25 steps)"), ("nmse", "Normalized MSE (lower is better)")]


def split(combo):
    enc, pred, loss = combo.split("-")
    return f"{enc}-{pred}", loss


def seed_dirs(combo_dir):
    return sorted((d for d in combo_dir.glob("seed*") if (d / "metrics.csv").exists()),
                  key=lambda d: int(d.name[4:]))


def load_metrics(run_dir):
    df = pd.read_csv(run_dir / "metrics.csv")
    return df.drop(columns=[c for c in ("epoch",) if c in df]).groupby("step").first()


def band(frames, column, smooth):
    """frames: per-seed DataFrames -> (steps, mean, low, high) of the smoothed column, or None."""
    series = [f[column].dropna() for f in frames if column in f and f[column].notna().any()]
    if not series:
        return None
    table = pd.concat([s.rolling(smooth, min_periods=1).mean() for s in series], axis=1).sort_index()
    table = table.interpolate(limit_area="inside")
    return table.index.to_numpy(), table.mean(1).to_numpy(), table.min(1).to_numpy(), table.max(1).to_numpy()


def draw(ax, curve, color, label=None):
    steps, mean, low, high = curve
    if len(steps) > 1:
        ax.fill_between(steps, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(steps, mean, color=color, label=label)
    else:
        ax.plot(steps, mean, "o", color=color, markersize=6, label=label)


def autoscale(ax, curves):
    values = np.concatenate([c[2] for c in curves] + [c[3] for c in curves])
    values = values[np.isfinite(values)]
    if len(values) and values.min() > 0 and values.max() / values.min() > 30:
        ax.set_yscale("log")


def default_smooth(frames):
    steps = max((len(f) for f in frames), default=1)
    return max(1, steps // 200)


def combo_report(combo_dir, smooth=None):
    combo_dir = Path(combo_dir)
    runs = seed_dirs(combo_dir)
    if not runs:
        return
    frames = [load_metrics(r) for r in runs]
    smooth = smooth or default_smooth(frames)
    arch, loss = split(combo_dir.name)
    color = COLORS.get(arch, COLORS["cnn-mlp"])
    columns = [c for c, _ in PANELS] + sorted({c for f in frames for c in f.columns} - {c for c, _ in PANELS})
    titles = dict(PANELS)
    curves = [(c, band(frames, c, smooth)) for c in columns]
    curves = [(c, b) for c, b in curves if b is not None]
    ncols = 4
    nrows = math.ceil(len(curves) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for ax, (column, curve) in zip(axes.flat, curves):
        draw(ax, curve, color)
        autoscale(ax, [curve])
        title = titles.get(column, column)
        other = "lewm/" if loss == "ref" else "ref/"
        if other in column:
            title += " (monitored, not trained)"
        ax.set_title(title)
        ax.set_xlabel("step")
    for ax in axes.flat[len(curves):]:
        ax.set_visible(False)
    fig.suptitle(f"{combo_dir.name}: mean of {len(runs)} seed(s), band = min to max across seeds, "
                 f"rolling mean over {smooth} step(s)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    (combo_dir / "plots").mkdir(exist_ok=True)
    fig.savefig(combo_dir / "plots" / "training.png", dpi=110)
    plt.close(fig)
    rows = [run_row(r) for r in runs]
    (combo_dir / "summary.json").write_text(json.dumps({"runs": rows, "mean": aggregate(rows)}, indent=2,
                                                       default=_json) + "\n")


def _json(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def anchors(results_dir):
    a = {k: read_json(results_dir / "anchors" / k / "proxy.json") for k in ("floor", "ref_holdout")}
    if not all(a.values()):
        return None
    return {k: next(iter(v.values())) for k, v in a.items()}


def run_row(run_dir, anchor=None):
    config = read_json(run_dir / "config.json") or {}
    done = read_json(run_dir / "done.json") or {}
    row = {"combo": run_dir.parent.name, "seed": config.get("seed"), "train_minutes": None, "onnx_mib": None,
           "test_submission_passed": None}
    if config:
        row.update(train_minutes=round(config["train_seconds"] / 60, 1), onnx_mib=config.get("onnx_mib"),
                   parameters=sum(config["parameters"].values()))
    codes = done.get("eval_exit_codes") or {}
    if "test_submission" in codes:
        row["test_submission_passed"] = codes["test_submission"] == 0
    proxy = read_json(run_dir / "proxy.json")
    if proxy:
        r = next(iter(proxy.values()))
        for key in ("top1", "nmse", "encoder_only", "pixel_r2", "pos_r2", "pixel_r2_enc", "pos_r2_enc"):
            row[key] = r.get(key)
        for step in r.get("steps", []):
            row[f"top1_k{step['k']}"] = step["top1"]
        if anchor:
            scaled = [(row[k] - anchor["floor"][k]) / (anchor["ref_holdout"][k] - anchor["floor"][k])
                      for k in ("top1", "pos_r2") if row.get(k) is not None]
            row["blend"] = float(np.mean(scaled)) if scaled else None
    action = read_json(run_dir / "action.json")
    if action:
        a = next(iter(action.values()))
        for horizon in ("20", "25"):
            if horizon in a:
                row[f"action_top1_{horizon}"] = a[horizon]["action_top1_of_8"]
                row[f"action_mrr_{horizon}"] = a[horizon]["action_mrr_of_8"]
                row[f"beats_reordered_{horizon}"] = a[horizon]["recorded_order_beats_reordered"]
    if (run_dir / "metrics.csv").exists():
        m = load_metrics(run_dir)
        tail = m.iloc[-max(1, len(m) // 50):]  # last 2% of the logged steps
        for column in ("train/loss", "train/mse", "train/batch_top1"):
            if column in m:
                row[f"final_{column.split('/', 1)[1]}"] = float(tail[column].mean())
    return row


def aggregate(rows):
    df = pd.DataFrame(rows).select_dtypes("number").drop(columns=["seed"], errors="ignore")
    return {c: {"mean": df[c].mean(), "std": df[c].std(), "n": int(df[c].notna().sum())} for c in df}


def report(results_dir, smooth=None):
    results_dir = Path(results_dir)
    combo_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and d.name.count("-") == 2)
    for d in combo_dirs:
        combo_report(d, smooth)
    out = results_dir / "report"
    out.mkdir(exist_ok=True)
    anchor = anchors(results_dir)
    rows = [run_row(r, anchor) for d in combo_dirs for r in seed_dirs(d)]
    if not rows:
        print(f"no finished runs under {results_dir}")
        return
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    numeric = per_run.select_dtypes("number").columns.drop("seed", errors="ignore")
    summary = per_run.groupby("combo")[list(numeric)].agg(["mean", "std", "count"])
    summary.to_csv(out / "summary.csv")
    write_markdown(per_run, out / "summary.md", anchor is not None)
    training_comparison(combo_dirs, out, smooth)
    eval_comparison(per_run, out)
    horizon_comparison(per_run, out)
    print(f"wrote {out}")


def write_markdown(per_run, path, has_blend):
    cols = ["top1", "nmse", "encoder_only", "pos_r2"] + (["blend"] if has_blend else []) + \
           ["action_top1_25", "final_mse", "train_minutes", "onnx_mib", "test_submission_passed"]
    cols = [c for c in cols if c in per_run]
    lines = ["| combination | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for combo, g in per_run.groupby("combo"):
        cells = []
        for c in cols:
            v = g[c].dropna()
            if c == "test_submission_passed":
                cells.append(f"{int(v.sum())}/{len(v)}" if len(v) else "-")
            elif not len(v):
                cells.append("-")
            else:
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}")
        lines.append(f"| {combo} | {len(g)} | " + " | ".join(cells) + " |")
    path.write_text("Mean ± standard deviation across seeds.\n\n" + "\n".join(lines) + "\n")


def training_comparison(combo_dirs, out, smooth):
    data = {d.name: [load_metrics(r) for r in seed_dirs(d)] for d in combo_dirs}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return
    smooth = smooth or max(default_smooth(v) for v in data.values())
    fig, axes = plt.subplots(len(KEY_PANELS), 2, figsize=(11, 2.6 * len(KEY_PANELS)), sharey="row", squeeze=False)
    titles = dict(PANELS)
    for col, loss in enumerate(LOSSES):
        for row, column in enumerate(KEY_PANELS):
            ax = axes[row, col]
            curves = []
            for arch in ARCHS:
                frames = data.get(f"{arch}-{loss}")
                curve = band(frames, column, smooth) if frames else None
                if curve is not None:
                    draw(ax, curve, COLORS[arch], label=arch)
                    curves.append(curve)
            if curves:
                autoscale(ax, curves)
            ax.set_title(f"{titles[column]}" if row else f"{LOSS_TITLE[loss]}\n{titles[column]}")
            if row == len(KEY_PANELS) - 1:
                ax.set_xlabel("step")
    handles = [plt.Line2D([], [], color=COLORS[a], label=a) for a in ARCHS]
    fig.legend(handles=handles, loc="upper left", ncol=len(ARCHS), bbox_to_anchor=(0.01, 0.995))
    fig.text(0.99, 0.995, f"mean of seeds, band = min to max, rolling mean over {smooth} step(s)", ha="right",
             va="top", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "training_comparison.png", dpi=110)
    plt.close(fig)


def order(combos):
    rank = {f"{a}-{l}": i for i, (l, a) in enumerate((l, a) for l in LOSSES for a in ARCHS)}
    return sorted(combos, key=lambda c: rank.get(c, len(rank)))


def eval_comparison(per_run, out):
    metrics = [(k, t) for k, t in EVAL_METRICS if k in per_run and per_run[k].notna().any()]
    if not metrics:
        return
    combos = order(per_run["combo"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 0.42 * len(combos) + 1.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(combos))
    for ax, (key, title) in zip(axes[0], metrics):
        for i, combo in enumerate(combos):
            values = per_run.loc[per_run["combo"] == combo, key].dropna().to_numpy()
            if not len(values):
                continue
            color = COLORS.get(split(combo)[0], INK2)
            ax.barh(i, values.mean(), height=0.5, color=color, alpha=0.85)
            ax.plot(values, np.full(len(values), i), "o", color=INK, markersize=3.5, alpha=0.8)
            if len(values) > 1:
                ax.plot([values.min(), values.max()], [i, i], color=INK, linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(y, combos)
    axes[0, 0].invert_yaxis()
    fig.text(0.01, 0.005, "bar = mean over seeds, dots = individual seeds, line = range", color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "eval_comparison.png", dpi=110)
    plt.close(fig)


def horizon_comparison(per_run, out):
    cols = [f"top1_k{k}" for k in range(1, 6)]
    if not all(c in per_run for c in cols):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    x = np.arange(1, 6) * 5
    for ax, loss in zip(axes, LOSSES):
        for arch in ARCHS:
            g = per_run[per_run["combo"] == f"{arch}-{loss}"][cols].dropna()
            if g.empty:
                continue
            ax.fill_between(x, g.min().to_numpy(), g.max().to_numpy(), color=COLORS[arch], alpha=0.18, linewidth=0)
            ax.plot(x, g.mean().to_numpy(), "-o", color=COLORS[arch], markersize=4, label=arch)
        ax.set_title(LOSS_TITLE[loss])
        ax.set_xlabel("environment steps ahead")
        ax.set_xticks(x)
    axes[0].set_ylabel("held-out retrieval top-1")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "top1_by_horizon.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="default", help="experiments/ablations/results/<name>/")
    p.add_argument("--smooth", type=int, default=None, help="rolling-mean window in steps (default: run length/200)")
    args = p.parse_args()
    report(RESULTS / args.name, args.smooth)


if __name__ == "__main__":
    main()
