"""Combinations of the InfoNCE experiments: suites, report panels and colors. Plain data, no torch.

Names are the ablation's "<encoder>-<predictor>-<loss>". The loss presets themselves are registered in
experiments/ablations/presets.py (INFONCE_PRESETS), so that the shared train.py accepts them; import this module with
experiments/ablations on sys.path (launch.py and report.py here do that).
"""

from presets import MSE_GRID, NCE_GRID, grid_name

# Weight grid: InfoNCE weight in NCE_GRID x MSE weight in MSE_GRID, everything else as ref, on the reference
# architecture (CNN encoder, MLP predictor).
WEIGHTS = [f"cnn-mlp-{grid_name(n, m)}" for m in MSE_GRID for n in NCE_GRID]

SUITES = {
    # ref (the baseline) first, then by MSE weight and InfoNCE weight. ref and the grid's points at MSE 25 with InfoNCE
    # 2 and 0.5 (ref_nce2x, ref_nce05x) are already trained in the ablations' loss suite and get linked.
    "weights": ["cnn-mlp-ref"] + WEIGHTS,
}
# experiments/ablations/results: ref and ref_nce2x (from main and loss) and the blend anchors
REUSE_FROM = ["main", "loss"]

# Report figures: one panel per MSE weight, each with ref as the baseline. Color = InfoNCE weight (ref blue),
# line style = MSE weight (25 solid, 12 dashed, 6 dotted).
LOSS_STYLE = {
    "ref": (1, "-"),
    **{grid_name(n, m): (slot, style) for n, slot in zip(NCE_GRID, (6, 2, 3, 4, 5, 7))
       for m, style in zip(MSE_GRID, ("-", "--", ":"))},
}
LOSS_GROUPS = {f"InfoNCE weight at MSE weight {m}": ["ref"] + [grid_name(n, m) for n in NCE_GRID] for m in MSE_GRID}
