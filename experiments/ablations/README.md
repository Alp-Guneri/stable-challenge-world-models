# Encoder × predictor × loss ablation

Which part of LeWM versus the organizers' reference model makes the difference? Every encoder is paired with every
predictor and trained with each loss, all else held fixed.

| | Reference ([example_train.py](../../example_train.py), `ref_long`) | LeWM ([lewm.py](../../lewm.py), `lewm_long`) |
|---|---|---|
| Encoder | `cnn`: CNN over the 3 stacked frames, D = 256 | `vit`: per-frame ViT (D = 96, 6 layers, 4×24 heads, 8×8 patches) + BatchNorm projector |
| Predictor | `mlp`: residual MLP, hidden 512, raw actions | `ar`: causal transformer (6 layers), AdaLN-zero actions, BatchNorm projector |
| Loss | `ref`: 25·MSE + VCReg + VCReg within clip + InfoNCE | `lewm`: MSE + 0.1·SIGReg |

Nine combinations: the 2 × 2 × 2 grid plus `vit-mlp1-ref` (the literal reference MLP on the newest frame embedding
only). Each is named `<encoder>-<predictor>-<loss>`, e.g. `vit-ar-lewm`. `cnn-mlp-ref` is exactly the reference
model and `vit-ar-lewm` exactly `lewm_long` (same parameter count and the same outputs given the same weights).

## How mismatched pieces are joined

The CNN embeds a 3-frame clip (one embedding already contains velocity); the ViT embeds single frames (three
embeddings needed for velocity). [models.py](models.py) puts each encoder and predictor behind one interface; its
docstring has the details. In short:

- **Context.** The model starts from 1 clip embedding (`cnn`) or 3 frame embeddings (`vit`), and `z[:, 0]` is the
  newest. A shared rollout predicts, appends, and keeps the predictor's window.
- **`mlp` with `vit`** is a *history MLP*: the reference network fed the last 3 embeddings concatenated (a 3× wider
  first layer), residual on the newest. With `cnn` its window is 1, i.e. the reference predictor. **`mlp1`** keeps
  the window at 1 with `vit`, so it cannot see velocity: `vit-mlp1-ref` vs `vit-mlp-ref` measures what history adds.
- **`ar` with `cnn`** has one real embedding to start from, so its window grows 1 → 2 → 3 with predictions, then
  slides. Every position then has a real action; LeWM's action-mask token (for the actions before t, which the
  evaluator does not give) is only used with `vit`.
- **Embedding size** is set by the encoder; each predictor keeps its own size and adapts only its input and output.

## Optimizer settings

Learning-rate settings belong to the **network**, not the loss: each network keeps those of the model it comes from,
in its own AdamW optimizer (stable-pretraining assigns parameters to optimizers by module name; `models.OPTIM`):

| Network | lr | weight decay | schedule | gradient clipping |
|---|---|---|---|---|
| `cnn` encoder, `mlp`/`mlp1` predictor (reference) | 1e-3 | 1e-4 | cosine to 0 | none |
| `vit` encoder + projector, `ar` predictor + action embedder + mask + projector (LeWM) | 2e-4 | 1e-3 | 5% linear warmup, then cosine to 0 | norm 1.0 |

So the pure combinations train exactly as the originals, and a mixed one (e.g. `cnn-ar`) trains each half with its
own settings. Switching the loss changes only the loss terms. Both optimizers step on every batch from one backward
pass of the shared loss.

## Shared settings

32,768 steps (`--steps`), batch 128, float32 on any GPU node (`--precision`, `--constraint`; `ref_long` and
`lewm_long` used bf16 mixed precision, which needs Ampere-or-newer GPUs; float32 is kept the same on every node),
10% of episodes held out (`--val-frac 0.1`, the last 333, as [proxy_eval.py](../../proxy_eval.py)), 3 seeds
(`--seeds`; seed `k` sets the weights and the order windows are sampled in, not the split).

## Running it

```bash
# quick check of the whole pipeline and the plots: 10 steps per seed (minutes once the jobs start)
.venv/bin/python experiments/ablations/launch.py --name smoke --steps 10 --anchor-steps 10

# the real run: 9 jobs (3 seeds each, in series) + 1 anchor job, in parallel; a report job after all of them
.venv/bin/python experiments/ablations/launch.py --name main
```

Use `.venv/bin/python` (or `uv run --no-sync`), not plain `uv run`: `uv run` first re-syncs the venv to
`uv.lock`, which now pins CUDA 13 builds of PyTorch.

`--cpu` runs everything on the CPU partition instead (no GPU; training falls back to float32, so it is much
slower). Other entry points: `train.py` (one combination, locally or by hand), `anchors.py`, `report.py` (re-plot at any time,
e.g. `--smooth 1` for raw curves), and `launch.py --dry-run` / `--combos ...` / `--seeds N`.

## Suites and loss presets

`launch.py --suite` picks the list of combinations (`--combos` gives an explicit list instead); both are defined in
[presets.py](presets.py):

| suite | combinations |
|---|---|
| `main` (default) | the encoder × predictor × loss grid above, plus `vit-mlp1-ref` |
| `loss` | `cnn-mlp` (the best pair in `main`) with every loss preset |
| `encoder` | every ViT variant with the MLP predictor and the reference loss, plus `cnn-mlp-ref`, `vit-mlp-ref`, `vit-mlp1-ref` (see [Encoder suite](#encoder-suite)) |

```bash
# the loss suite; reuses main's finished cnn-mlp-ref, cnn-mlp-lewm and anchors instead of retraining them
.venv/bin/python experiments/ablations/launch.py --name loss --suite loss --reuse-from main
```

Every loss is the same weighted sum, `mse·MSE + infonce·InfoNCE(τ) + vcreg·VCReg + vcreg_t·VCReg(within clip) +
sigreg·SIGReg`; a preset only sets the weights. Combination names split on `-`, so presets use `_`.

| preset | MSE | InfoNCE (τ) | VCReg | VCReg within clip | SIGReg | question |
|---|---|---|---|---|---|---|
| `ref` | 25 | 1 (0.1) | 1 | 1 | 0 | the reference loss |
| `lewm` | 1 | 0 | 0 | 0 | 0.1 | the LeWM loss |
| `ref_nonce` | 25 | 0 | 1 | 1 | 0 | is InfoNCE what makes the reference loss better? |
| `ref_novcregt` | 25 | 1 (0.1) | 1 | 0 | 0 | does the within-clip VCReg term matter? |
| `lewm_nce` | 1 | 1 (0.1) | 0 | 0 | 0.1 | does InfoNCE alone close the gap to `ref`? |
| `ref_nce2x`, `ref_nce05x` | 25 | 2 / 0.5 (0.1) | 1 | 1 | 0 | is the InfoNCE weight under-tuned? |
| `ref_tau005`, `ref_tau02` | 25 | 1 (0.05 / 0.2) | 1 | 1 | 0 | is the InfoNCE temperature under-tuned? |

All five terms are computed and logged on every step whatever their weight (weight-0 terms without gradient), so
every run's curves are comparable; `ref/infonce` is always logged at τ = 0.1, and the temperature presets also log
`infonce_trained_tau`. `ref` and `lewm` give exactly the losses of the main suite (checked), so runs can be shared
between suites: `--reuse-from NAME` links each combination that results/NAME finished with the same `--steps` and
`--seeds` (and its anchors) instead of submitting it. The report adds `training_by_loss_<pair>.png` and
`top1_by_horizon_by_loss_<pair>.png` for any pair run with 3 or more presets, grouped as in `presets.LOSS_GROUPS`;
color follows the preset, and the two settings of one knob share a color (solid = higher, dashed = lower).

## Encoder suite

Why does the ViT reach ~22% held-out top-1 where the CNN reaches 73%? The two differ in how the frames enter (the
CNN sees the 3 frames stacked together; LeWM's ViT embeds each frame on its own) and in the architecture. Every
variant here changes one thing from `vit`, everything else as `lewm_long`, with the MLP predictor and the reference
loss (the best pair and loss in `main`; the ViT barely reacted to the loss there):

```bash
.venv/bin/python experiments/ablations/launch.py --name encoder --suite encoder --reuse-from main --time 3-00:00:00
```

| encoder | change from `vit` | tokens per clip | question |
|---|---|---|---|
| `vit` (baseline, reused) | LeWM's per-frame ViT: D 96, 6 layers, 4 heads, 8×8 patches, BatchNorm projector, lr 2e-4 | 3 × 65 | |
| `vitstack` | the 3 frames stacked as 9 channels before the patches; one embedding of the clip | 65 | is seeing the frames together what the CNN has? |
| `vitst` | one token per patch per frame + temporal embeddings, joint space-time attention; one embedding of the clip | 193 | does attention across frames add to that? |
| `vit_lr5e5`, `vit_lr1e3` | encoder learning rate 5e-5 (LeWM authors' default) / 1e-3 (the CNN's) | 3 × 65 | is 2e-4 a good rate? (also removes the CNN-ViT rate difference) |
| `vit_p4`, `vit_p16` | 4×4 / 16×16 patches | 3 × 257 / 3 × 17 | does an 8×8 patch blur the agent? |
| `vit_d192`, `vit_d48` | embedding size 192 / 48 (heads stay 4; head size, MLP and projector widths scale with D) | 3 × 65 | is D = 96 too little capacity? |
| `vit_ln` | LayerNorm instead of BatchNorm in the projector | 3 × 65 | does the BatchNorm projector hurt? |

Controls, reused from `main` by `--reuse-from main`: `vit-mlp-ref` (the baseline every variant is compared with),
`cnn-mlp-ref` (the target: how much of the 22% → 73% gap a variant closes) and `vit-mlp1-ref` (the per-frame ViT
with no predictor history). The clip encoders (`vitstack`, `vitst`) give one embedding per 3-frame window, like the
CNN, so their MLP predictor sees one embedding (exactly the reference predictor); the per-frame variants give 3, and
the MLP sees all 3 (the history MLP of `vit-mlp`). Variants and their settings are in `presets.ENCODER_VARIANTS`.

Notes:
- `vit` is exactly `main`'s ViT (same parameter count, outputs equal given the same weights). The ViT encoders now
  compute attention with PyTorch's `scaled_dot_product_attention`, the same computation as LeWM's explicit
  softmax(QK^T/√d)V without storing the attention matrix: with 4×4 patches that matrix would need ~1 GB per layer
  per batch, more than the 8 GB GTX 1080s jobs often land on. Encoder attention is never causal.
- Cost: `vit_p4` is ~5× and `vitst` ~2–3× the baseline's training time, so allow `--time 3-00:00:00`. All variants
  pass `test_submission.py`; `vit_p4` takes ~2 min of the 4-minute CPU evaluation budget (on a 24-core node), so it
  is a diagnostic rather than a submission candidate as is.
- At lr 5e-5, 32,768 steps may not be enough to converge: check its curves for a plateau before reading its result.
- Report: `training_by_arch_ref.png` and `top1_by_horizon_by_arch_ref.png`, one panel per question
  (`presets.ARCH_GROUPS`), each with the CNN target and the ViT baseline. The two settings of one knob share a color
  (solid = larger value, dashed = smaller); `vit-mlp1` is `vit-mlp`'s color, dashed (same encoder).
- If a clip encoder clearly wins, the settings sweep is worth repeating on it: what helps a per-frame ViT need not
  transfer.

## Outputs

`experiments/ablations/results/<name>/` (git-ignored):

```
<combo>/seed<k>/
    metrics.csv            every training step: loss being trained, all terms of BOTH losses (the other one is
                           computed without gradient), prediction MSE per horizon, in-batch retrieval top-1,
                           cosine(pred, target), embedding std / effective rank, prediction std, gradient norm per
                           network (before clipping), learning rate per network, seconds per step
    config.json            settings, parameter counts, ONNX size, train time, GPU, host, git commit, Slurm job id
    model.onnx             the export (the submission format)
    weights.pt             PyTorch weights
    test_submission.txt    official format check output (pass/fail per check, speed estimate)
    proxy.json, proxy_eval.txt          held-out retrieval, nmse, probes, per horizon
    action.json, action_diagnostic.txt  does the model rank the recorded actions first? (out of 8)
    done.json              evaluation exit codes; a seed with this file is skipped on resubmission
    spt/                   stable-pretraining's run metadata and raw logger output
<combo>/plots/training.png   every metric above: mean of the seeds, band = min to max across seeds
<combo>/summary.json         evaluation results per seed and averaged
anchors/{floor,ref_holdout}/ the two models that define the local blend's 0 and 1, and their proxy results
report/                      training_comparison.png, eval_comparison.png, top1_by_horizon.png,
                             per_run.csv, summary.csv, summary.md (mean ± std per combination)
slurm/                       job logs;  jobs.json  the submitted job ids
```

## Limitations

- **Adapters.** `vit-mlp` (history MLP) and the growing window of `cnn-ar` are not exactly either original model.
- **Settings tuned for their own model.** Loss weights (25·MSE with unit regularizers; MSE + 0.1·SIGReg) and
  learning rates were tuned for their original pairing. In a mixed combination they may be off, so a weak result
  can reflect untuned settings rather than the piece itself. The reference loss's 25× MSE weight also makes its
  effective step size larger under LeWM's lower learning rate.
- **Gradient clipping is per network.** LeWM clips the norm of all its gradients together; here each network's are
  clipped separately (same threshold), so `vit-ar-lewm` differs from `lewm_long` in that one respect.
- **One step count for all.** 32,768 steps lies between `ref_long` (24k) and `lewm_long` (40k).
- **The local metrics favor the reference loss.** InfoNCE trains retrieval directly, and held-out top-1 measures
  retrieval. Read nmse, the probes and the action diagnostic alongside it. The blend is the team's local stand-in
  for the hidden score, anchored to its own 6k-step reference, not the leaderboard's.
- **Units differ between encoders.** MSE, gradient norms and embedding statistics are in each encoder's own units
  (D = 256 vs 96, different scales). Compare them within an encoder; across encoders use retrieval, nmse, the probes
  and the action diagnostic, which are scale-free.
- **Three seeds.** Seed-to-seed spread for the reference is about ±1.6 points of top-1; differences of that size are
  noise. The plotted bands are min to max because a standard deviation from 3 values is unreliable.
