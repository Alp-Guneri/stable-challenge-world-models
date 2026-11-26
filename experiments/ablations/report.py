"""Plots and tables for the ablation.

    .venv/bin/python experiments/ablations/report.py --name main             # every combination + comparison
    .venv/bin/python experiments/ablations/report.py --name main --smooth 1  # no smoothing

Per combination (written by train.py when its seeds finish, or here):
    <combo>/plots/training.png   every logged training metric: mean over seeds, band = min..max over seeds
    <combo>/summary.json         per-seed and aggregated evaluation results
Across combinations (results/<name>/report/):
    eval_comparison.png          held-out metrics per combination: mean (bar) and each seed (dots)
    per_run.csv, summary.csv, summary.md
  when several encoder-predictor pairs were run (e.g. the main suite):
    training_comparison.png      key training metrics, one column per original loss, one line per pair
    top1_by_horizon.png          held-out retrieval at 5..25 steps ahead, the same layout
  for each pair run with 3 or more loss presets (e.g. the loss suite):
    training_by_loss_<pair>.png          key training metrics, one column per group of presets.LOSS_GROUPS,
                                         one line per loss preset
    top1_by_horizon_by_loss_<pair>.png   held-out retrieval by horizon, the same layout

Curves are smoothed with a trailing rolling mean (default: 1/200 of the run) before the seeds are combined.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
log = logging.getLogger("ablations.report")
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
import presets  # noqa: E402

# Reference categorical palette, slots 1-8 (fixed order, never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Encoder-predictor pairs in a fixed order, each with a fixed color (slots 1-5).
ARCHS = presets.MAIN_ARCHS  # the main suite's pairs, in its figures' order
# Each pair's color and line style (presets.ARCH_STYLE); e.g. vit-mlp1 is vit-mlp's color, dashed (same encoder).
ARCH_COLORS = {k: PALETTE[slot - 1] for k, (slot, _) in presets.ARCH_STYLE.items()}
ARCH_LINES = {k: style for k, (_, style) in presets.ARCH_STYLE.items()}
COLORS = ARCH_COLORS
LOSSES = ["ref", "lewm"]  # the two original losses: the encoder x predictor figures compare these
LOSS_TITLE = {"ref": "Reference loss (MSE + VCReg + InfoNCE)", "lewm": "LeWM loss (MSE + SIGReg)"}
# Each loss preset's color and line style in the per-loss figures (presets.LOSS_STYLE).
LOSS_COLORS = {k: PALETTE[slot - 1] for k, (slot, _) in presets.LOSS_STYLE.items()}
LOSS_LINES = {k: style for k, (_, style) in presets.LOSS_STYLE.items()}
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
    ("train/emb_partial_std", "Partial (context-only) embedding std"),  # causal encoders only
    ("train/emb_partial_eff_rank", "Partial (context-only) embedding effective rank"),
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


def draw(ax, curve, color, label=None, linestyle="-"):
    steps, mean, low, high = curve
    if len(steps) > 1:
        ax.fill_between(steps, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(steps, mean, color=color, label=label, linestyle=linestyle)
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


def save(fig, path):
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log.info("  wrote %s", path)


def combo_report(combo_dir, smooth=None):
    combo_dir = Path(combo_dir)
    runs = seed_dirs(combo_dir)
    if not runs:
        return
    frames = [load_metrics(r) for r in runs]
    smooth = smooth or default_smooth(frames)
    arch, loss = split(combo_dir.name)
    color = ARCH_COLORS.get(arch, PALETTE[0])
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
        if not presets.trains(loss, column):
            title += " (monitored, not trained)"
        ax.set_title(title)
        ax.set_xlabel("step")
    for ax in axes.flat[len(curves):]:
        ax.set_visible(False)
    fig.suptitle(f"{combo_dir.name}: mean of {len(runs)} seed(s), band = min to max across seeds, "
                 f"rolling mean over {smooth} step(s)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    (combo_dir / "plots").mkdir(exist_ok=True)
    save(fig, combo_dir / "plots" / "training.png")
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
        gpu = (config.get("gpu") or "CPU").replace("NVIDIA ", "").replace("GeForce ", "")
        row.update(host=config.get("host"), gpu=gpu)
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


def report(results_dir, smooth=None, arch_style=None, arch_groups=None, combo_order=None):
    """arch_style ({pair: (slot, line style)}), arch_groups ({panel title: [pairs]}) and combo_order (combination
    names, in the order of the bar chart) let another experiment draw its own pairs (experiments/predictor_history);
    by default they come from presets.py."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # no-op if the caller configured logging
    results_dir = Path(results_dir)
    log.info("report for %s", results_dir)
    colors = {**ARCH_COLORS, **{k: PALETTE[slot - 1] for k, (slot, _) in (arch_style or {}).items()}}
    lines = {**ARCH_LINES, **{k: style for k, (_, style) in (arch_style or {}).items()}}
    combo_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and d.name.count("-") == 2)
    for d in combo_dirs:
        if not seed_dirs(d):
            log.warning("skipped %s: no seed has a metrics.csv yet", d.name)
        combo_report(d, smooth)
    out = results_dir / "report"
    out.mkdir(exist_ok=True)
    anchor = anchors(results_dir)
    rows = [run_row(r, anchor) for d in combo_dirs for r in seed_dirs(d)]
    if not rows:
        log.warning("no finished runs under %s", results_dir)
        return
    per_run = pd.DataFrame(rows)
    log_added(per_run, combo_order)
    per_run.to_csv(out / "per_run.csv", index=False)
    numeric = per_run.select_dtypes("number").columns.drop("seed", errors="ignore")
    summary = per_run.groupby("combo")[list(numeric)].agg(["mean", "std", "count"])
    summary.to_csv(out / "summary.csv")
    write_markdown(per_run, out / "summary.md", anchor is not None)
    names = [d.name for d in combo_dirs]
    pairs = {split(c)[0] for c in names}
    if len(pairs) > 1 and pairs <= set(ARCHS):  # the main suite's pairs: compare them per original loss
        training_comparison(combo_dirs, out, smooth)
        horizon_comparison(per_run, out)
    for loss, archs in arch_suites(names).items():  # pairs beyond the main suite's (encoder suite): grouped
        by_arch_figures(loss, archs, combo_dirs, per_run, out, smooth, arch_groups or presets.ARCH_GROUPS, colors,
                        lines)
    for arch, losses in loss_suites(names).items():  # one pair with many loss presets: compare the losses
        by_loss_figures(arch, losses, combo_dirs, per_run, out, smooth)
    eval_comparison(per_run, out, colors, combo_order)
    log.info("wrote %s", out)


def log_added(per_run, combo_order=None):
    """One line per combination in the report, in the bar chart's order: its seeds and held-out top-1."""
    log.info("combinations in the report:")
    for combo in order(per_run["combo"].unique(), combo_order):
        g = per_run[per_run["combo"] == combo]
        seeds = ", ".join(str(int(s)) for s in g["seed"].dropna())
        missing = int(g["top1"].isna().sum()) if "top1" in g else len(g)
        top1 = g["top1"].mean() if "top1" in g and g["top1"].notna().any() else None
        log.info("  added %s: %d seed(s) [%s]%s%s", combo, len(g), seeds,
                 f", top1 {top1:.3f}" if top1 is not None else "",
                 f" ({missing} without held-out evaluation)" if missing else "")


def write_markdown(per_run, path, has_blend):
    cols = ["top1", "nmse", "encoder_only", "pos_r2"] + (["blend"] if has_blend else []) + \
           ["action_top1_25", "final_mse", "train_minutes", "onnx_mib", "test_submission_passed"]
    cols = [c for c in cols if c in per_run]
    header = ["train_minutes (min–max)" if c == "train_minutes" else c for c in cols]
    header += ["node (GPU)"] if "host" in per_run else []
    lines = ["| combination | seeds | " + " | ".join(header) + " |", "|" + "---|" * (len(header) + 2)]
    for combo, g in per_run.groupby("combo"):
        cells = []
        for c in cols:
            v = g[c].dropna()
            if c == "test_submission_passed":
                cells.append(f"{int(v.sum())}/{len(v)}" if len(v) else "-")
            elif not len(v):
                cells.append("-")
            elif c == "train_minutes":  # training time of one seed: mean, and the fastest and slowest seed
                cells.append(f"{v.mean():.1f} ({v.min():.1f}–{v.max():.1f})" if len(v) > 1 else f"{v.mean():.1f}")
            else:
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}")
        if "host" in g:  # where the seeds ran; seeds of one job share a node, a resubmitted job may not
            places = g[["host", "gpu"]].dropna().drop_duplicates()
            cells.append(", ".join(f"{h} ({u})" for h, u in places.itertuples(index=False)) or "-")
        lines.append(f"| {combo} | {len(g)} | " + " | ".join(cells) + " |")
    path.write_text("Mean ± standard deviation across seeds; train_minutes is the mean training time of one seed with "
                    "the fastest and slowest seed in brackets (evaluation not included).\n\n" + "\n".join(lines) + "\n")


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
                    draw(ax, curve, COLORS[arch], label=arch, linestyle=ARCH_LINES[arch])
                    curves.append(curve)
            if curves:
                autoscale(ax, curves)
            ax.set_title(f"{titles[column]}" if row else f"{LOSS_TITLE[loss]}\n{titles[column]}")
            if row == len(KEY_PANELS) - 1:
                ax.set_xlabel("step")
    handles = [plt.Line2D([], [], color=COLORS[a], linestyle=ARCH_LINES[a], label=a) for a in ARCHS]
    fig.legend(handles=handles, loc="upper left", ncol=len(ARCHS), bbox_to_anchor=(0.01, 0.995))
    fig.text(0.99, 0.995, f"mean of seeds, band = min to max, rolling mean over {smooth} step(s)", ha="right",
             va="top", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, out / "training_comparison.png")


def order(combos, combo_order=None):
    """In combo_order if given (others last); otherwise grouped by loss (presets.py order), then by encoder-predictor
    pair (ARCHS order)."""
    if combo_order:
        return sorted(combos, key=lambda c: (combo_order.index(c) if c in combo_order else len(combo_order), c))

    def key(c):
        arch, loss = split(c)
        return (presets.LOSSES.index(loss) if loss in presets.LOSSES else len(presets.LOSSES),
                ARCHS.index(arch) if arch in ARCHS else len(ARCHS), c)
    return sorted(combos, key=key)


def combo_color(combo, single_arch, colors=None):
    """Color by encoder-predictor pair; by loss preset when every combination shares one pair."""
    arch, loss = split(combo)
    return LOSS_COLORS.get(loss, INK2) if single_arch else (colors or ARCH_COLORS).get(arch, INK2)


def loss_suites(combos):
    """{arch: losses present} for every encoder-predictor pair run with 3 or more loss presets."""
    by_arch = {}
    for c in combos:
        arch, loss = split(c)
        by_arch.setdefault(arch, []).append(loss)
    return {a: ls for a, ls in by_arch.items() if len(ls) >= 3}


def arch_suites(combos):
    """{loss: pairs present} for every loss run with 2 or more encoder-predictor pairs, when some pair is outside the
    main suite's five (e.g. the encoder suite); the main suite has its own figures (training_comparison)."""
    by_loss = {}
    for c in combos:
        arch, loss = split(c)
        by_loss.setdefault(loss, []).append(arch)
    if all(a in ARCHS for archs in by_loss.values() for a in archs):
        return {}
    return {l: archs for l, archs in by_loss.items() if len(archs) >= 2}


def groups_for(present, preset_groups, other="Other"):
    """preset_groups restricted to the series present (groups with fewer than 2 dropped); the rest in `other`."""
    groups = {t: [x for x in xs if x in present] for t, xs in preset_groups.items()}
    groups = {t: xs for t, xs in groups.items() if len(xs) > 1}
    rest = [x for x in present if not any(x in xs for xs in groups.values())]
    if rest:
        groups[other] = rest
    return groups


def grouped_training(series, groups, colors, lines, title, filename, out, smooth):
    """Key training metrics: one column per group, one line per series. series: {label: combination folder}."""
    data = {k: [load_metrics(r) for r in seed_dirs(d)] for k, d in series.items()}
    data = {k: v for k, v in data.items() if v}
    groups = {t: [x for x in xs if x in data] for t, xs in groups.items()}
    groups = {t: xs for t, xs in groups.items() if xs}
    if not groups:
        return
    smooth = smooth or max(default_smooth(v) for v in data.values())
    panels = KEY_PANELS
    fig, axes = plt.subplots(len(panels), len(groups), figsize=(5 * len(groups), 2.6 * len(panels)),
                             sharey="row", squeeze=False)
    titles = dict(PANELS)
    for col, (group, members) in enumerate(groups.items()):
        for row, column in enumerate(panels):
            ax = axes[row, col]
            curves = []
            for key in members:
                curve = band(data[key], column, smooth)
                if curve is not None:
                    draw(ax, curve, colors.get(key, INK2), label=key, linestyle=lines.get(key, "-"))
                    curves.append(curve)
            if curves:
                autoscale(ax, curves)
            ax.set_title(titles[column] if row else f"{group}\n{titles[column]}")
            if row == 0:
                ax.legend(loc="best", fontsize=7.5)
            if row == len(panels) - 1:
                ax.set_xlabel("step")
    fig.suptitle(f"{title} (mean of seeds, band = min to max, rolling mean over {smooth} step(s))",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out / filename)


def grouped_horizon(series, groups, colors, lines, title, filename, per_run, out):
    """Held-out top-1 by horizon: one panel per group, one line per series. series: {label: combination name}."""
    cols = [f"top1_k{k}" for k in range(1, 6)]
    if not all(c in per_run for c in cols):
        return
    rows = {k: per_run[per_run["combo"] == c][cols].dropna() for k, c in series.items()}
    rows = {k: g for k, g in rows.items() if not g.empty}
    groups = {t: [x for x in xs if x in rows] for t, xs in groups.items()}
    groups = {t: xs for t, xs in groups.items() if xs}
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.6), sharey=True, squeeze=False)
    x = np.arange(1, 6) * 5
    for ax, (group, members) in zip(axes[0], groups.items()):
        for key in members:
            g, color = rows[key], colors.get(key, INK2)
            ax.fill_between(x, g.min().to_numpy(), g.max().to_numpy(), color=color, alpha=0.15, linewidth=0)
            ax.plot(x, g.mean().to_numpy(), marker="o", linestyle=lines.get(key, "-"), color=color, markersize=4,
                    label=key)
        ax.set_title(f"{title}: {group}")
        ax.set_xlabel("environment steps ahead")
        ax.set_xticks(x)
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0, 0].set_ylabel("held-out retrieval top-1")
    fig.tight_layout()
    save(fig, out / filename)


def by_loss_figures(arch, losses, combo_dirs, per_run, out, smooth):
    """One encoder-predictor pair, many loss presets (e.g. the loss suite)."""
    groups = groups_for(losses, presets.LOSS_GROUPS, "Other losses")
    dirs = {split(d.name)[1]: d for d in combo_dirs if split(d.name)[0] == arch}
    grouped_training({l: dirs[l] for l in losses if l in dirs}, groups, LOSS_COLORS, LOSS_LINES,
                     f"{arch}: loss presets", f"training_by_loss_{arch}.png", out, smooth)
    grouped_horizon({l: f"{arch}-{l}" for l in losses}, groups, LOSS_COLORS, LOSS_LINES, arch,
                    f"top1_by_horizon_by_loss_{arch}.png", per_run, out)


def by_arch_figures(loss, archs, combo_dirs, per_run, out, smooth, arch_groups=None, colors=None, lines=None):
    """One loss, many encoder-predictor pairs beyond the main suite's (e.g. the encoder suite)."""
    colors, lines = colors or ARCH_COLORS, lines or ARCH_LINES
    groups = groups_for(archs, arch_groups or presets.ARCH_GROUPS, "Other encoder-predictor pairs")
    dirs = {split(d.name)[0]: d for d in combo_dirs if split(d.name)[1] == loss}
    grouped_training({a: dirs[a] for a in archs if a in dirs}, groups, colors, lines,
                     f"loss {loss}: encoder-predictor pairs", f"training_by_arch_{loss}.png", out, smooth)
    grouped_horizon({a: f"{a}-{loss}" for a in archs}, groups, colors, lines, f"loss {loss}",
                    f"top1_by_horizon_by_arch_{loss}.png", per_run, out)


def eval_comparison(per_run, out, colors=None, combo_order=None):
    metrics = [(k, t) for k, t in EVAL_METRICS if k in per_run and per_run[k].notna().any()]
    if not metrics:
        return
    combos = order(per_run["combo"].unique(), combo_order)
    single_arch = len({split(c)[0] for c in combos}) == 1
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 0.42 * len(combos) + 1.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(combos))
    for ax, (key, title) in zip(axes[0], metrics):
        for i, combo in enumerate(combos):
            values = per_run.loc[per_run["combo"] == combo, key].dropna().to_numpy()
            if not len(values):
                continue
            color = combo_color(combo, single_arch, colors)
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
    save(fig, out / "eval_comparison.png")


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
            ax.plot(x, g.mean().to_numpy(), marker="o", linestyle=ARCH_LINES[arch], color=COLORS[arch], markersize=4,
                    label=arch)
        ax.set_title(LOSS_TITLE[loss])
        ax.set_xlabel("environment steps ahead")
        ax.set_xticks(x)
    axes[0].set_ylabel("held-out retrieval top-1")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out / "top1_by_horizon.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="default", help="<results-root>/<name>/")
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--smooth", type=int, default=None, help="rolling-mean window in steps (default: run length/200)")
    args = p.parse_args()
    report(args.results_root / args.name, args.smooth)


if __name__ == "__main__":
    main()
