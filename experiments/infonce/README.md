# InfoNCE experiments

Experiments on the InfoNCE term of the reference loss. The ablations' loss suite found that InfoNCE accounts for most
of the reference loss's advantage: removing it drops held-out top-1 from 73.0% to 38.8%. Halving or doubling its
weight, or its temperature, moved top-1 by at most about 1 point. All runs here use the reference architecture (CNN
encoder, MLP predictor).

## Weight grid (`--name weights`)

How should the MSE and InfoNCE terms be balanced? Every pair of

- InfoNCE weight 0.5, 2, 4, 8, 16, 32
- MSE weight 25, 12, 6

with everything else as in the reference loss (`ref`: MSE 25, InfoNCE 1 at τ = 0.1, VCReg 1, VCReg within clip 1).

| | InfoNCE 0.5 | InfoNCE 2 | InfoNCE 4 | InfoNCE 8 | InfoNCE 16 | InfoNCE 32 |
|---|---|---|---|---|---|---|
| **MSE 25** | `ref_nce05x` † | `ref_nce2x` † | `ref_nce4x` | `ref_nce8x` | `ref_nce16x` | `ref_nce32x` |
| **MSE 12** | `ref_nce05x_mse12` | `ref_nce2x_mse12` | `ref_nce4x_mse12` | `ref_nce8x_mse12` | `ref_nce16x_mse12` | `ref_nce32x_mse12` |
| **MSE 6** | `ref_nce05x_mse6` | `ref_nce2x_mse6` | `ref_nce4x_mse6` | `ref_nce8x_mse6` | `ref_nce16x_mse6` | `ref_nce32x_mse6` |

† Already trained in the ablations' loss suite, so these two and `ref` (the baseline, InfoNCE 1 at MSE 25) are
linked, not retrained: 16 combinations are new. Each is `cnn-mlp-<preset>`. A decimal weight drops its point in the
name (0.5 → `05`).

The CNN and MLP are optimized with Adam and no gradient clipping, so scaling every weight together changes little.
What the grid varies is the MSE : InfoNCE balance and how strong both are next to VCReg, whose weight stays 1. For
example, MSE 12 + InfoNCE 2 has the same MSE : InfoNCE ratio as MSE 25 + InfoNCE 4, with VCReg relatively twice as
strong.

## Running it

```bash
# print every sbatch command, submit nothing
.venv/bin/python experiments/infonce/launch.py --name weights --dry-run

# the real run: 16 jobs (3 seeds each, in series), then a report job
.venv/bin/python experiments/infonce/launch.py --name weights

# re-plot at any time
.venv/bin/python experiments/infonce/report.py --name weights
```

Every flag of [the ablation launcher](../ablations/launch.py) works here (`--seeds`, `--steps`, `--combos`,
`--constraint`, `--time`, `--no-reuse`, ...). `ref`, `ref_nce05x`, `ref_nce2x` and the blend anchors are linked from
`experiments/ablations/results/{main,loss}` only when those runs used the same `--steps` and `--seeds`.

## Results

Each run goes to `results/<name>/`, with the same files as the ablations. The report in `results/<name>/report/` has
`summary.md`, `summary.csv`, `per_run.csv`, `eval_comparison.png`, and:

- **`training_by_loss_cnn-mlp.png`:** key training curves, one column per MSE weight, each with `ref`.
- **`top1_by_horizon_by_loss_cnn-mlp.png`:** held-out top-1 at 5–25 steps ahead, the same layout.

Color is the InfoNCE weight (`ref` is blue) and line style the MSE weight: 25 solid, 12 dashed, 6 dotted. Each
panel has 7 lines: `ref` and the 6 InfoNCE weights.

## Where the code is

Training, evaluation, the loss and the report are shared with the ablations. The loss presets are registered in
[presets.py](../ablations/presets.py) (`INFONCE_PRESETS`, `NCE_GRID`, `MSE_GRID`), because the shared `train.py`
only accepts presets defined there. This folder holds the suites and plot styling
([infonce_presets.py](infonce_presets.py)), thin wrappers around the ablation's launcher and report, and the results.
To add an InfoNCE experiment: register its presets next to `INFONCE_PRESETS`, then add a suite (and panels) in
`infonce_presets.py`.
