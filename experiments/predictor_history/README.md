# Predictor history

Does giving the **predictor** history help when the **encoder** already sees several frames?

The ablations found that history matters most when the encoder sees it. Swapping the per-frame ViT for one that sees
all 3 frames (`vitst`) took held-out top-1 from 22% to 69%. Giving the per-frame ViT's predictor 3 embeddings instead
of 1 did not raise top-1 (22.1% vs 24.9%), though it did help motion R². Every strong model so far has an encoder
that sees 3 frames and a predictor that sees 1 embedding. This experiment gives history to both, in three ways.

The evaluator only provides 3 frames (t−10, t−5, t), so each option splits those frames differently between the
encoder and the predictor.

| Option | Encoder sees | Predictor sees |
|---|---|---|
| Baseline (`cnn-mlp-ref`, `vitst-mlp-ref`) | 3 frames, 1 embedding | 1 embedding |
| **1. Shorter clips** (`*_clip2`) | 2-frame clips: (t−10, t−5), (t−5, t) | window of 2: both real, then sliding |
| **2. Causal encoder** (`*_causal`, `*_causalsep`) | [t−10], [t−10, t−5], [t−10, t−5, t] | window of 3: all real, then sliding |
| **3. Predictor's own history** (`mlp_w3`, `mlp_w5`, `ar`, `ar_w5`) | 3 frames, 1 embedding (as baseline) | grows with its predictions up to 3 or 5 |

Every combination uses the reference loss, the reference CNN or the spacetime ViT (`vitst`), and either the
reference MLP or LeWM's AR transformer as predictor. That gives 20 combinations plus the 2 baselines:

| | CNN | ViT (`vitst`) |
|---|---|---|
| Baseline | `cnn-mlp-ref` † | `vitst-mlp-ref` † |
| Option 1 | `cnn_clip2-mlp-ref`, `cnn_clip2-ar_w2-ref` | `vitst_clip2-mlp-ref`, `vitst_clip2-ar_w2-ref` |
| Option 2, one network | `cnn_causal-mlp-ref`, `cnn_causal-ar-ref` | `vitst_causal-mlp-ref`, `vitst_causal-ar-ref` |
| Option 2, separate networks | `cnn_causalsep-mlp-ref`, `cnn_causalsep-ar-ref` | `vitst_causalsep-mlp-ref`, `vitst_causalsep-ar-ref` |
| Option 3, cap 3 | `cnn-mlp_w3-ref`, `cnn-ar-ref` † | `vitst-mlp_w3-ref`, `vitst-ar-ref` |
| Option 3, cap 5 | `cnn-mlp_w5-ref`, `cnn-ar_w5-ref` | `vitst-mlp_w5-ref`, `vitst-ar_w5-ref` |

† Already trained in the ablations (`main`, `encoder`). These are linked, not retrained, so 19 combinations are
new. Compare the CNN rows with `cnn-mlp-ref` and the ViT rows with `vitst-mlp-ref`.

## Design decisions

- **Option 1, both predictors use a window of 2.** The predictor sees exactly 2 embeddings, matching the encoder's 2-frame clips.
- **Option 2, one network (`causal`).**
  - **CNN:** the missing (oldest) frames are filled with zeros, and one 0/1 "frame present" channel per slot tells the network which frames are real. The network is otherwise the reference CNN.
  - **ViT:** the missing frames' tokens are left out. Time embeddings are aligned to the newest frame, so the newest frame always gets the same one. A single projector is applied to partial and full embeddings in one call, so its BatchNorm statistics cover the same mix in training and evaluation.
- **Option 2, separate networks (`causalsep`).** There is one encoder (and, for the ViT, one projector) for each number of frames. The 3-frame encoder is the baseline encoder. This variant has about 3× the encoder parameters of `causal`.
- **Option 2, regularizers on full-window embeddings only.** VCReg and SIGReg see only the 6 embeddings built from 3 full frames, the same set as the baseline. The 2 partial embeddings are context only, never targets, so they can't collapse in a way that games the loss. They are logged separately as `train/emb_partial_std` and `train/emb_partial_eff_rank`. A rank near 1 would mean the predictor ignores them, and then that option did not really test the idea.
- **Option 3, MLP with a growing window.** The MLP has a fixed input of 3 or 5 slots. Empty (oldest) slots are zero-filled, and one 0/1 "filled" input per slot says which slots are real. The AR predictor handles a growing window natively. Cap 5 never slides (1 context embedding + 4 predictions), so the real embedding stays in view for the whole rollout. Caps 3 and 5 only differ from step 4 on.
- **Missing past actions.** The evaluator gives no actions before t. The AR predictor uses LeWM's learned mask token for positions before t: 1 in option 1, 2 in option 2, none in option 3. The MLP only reads the newest action.
- **Everything else as in the ablations.** That is 3 seeds × 32,768 steps, batch 128, float32, one optimizer per network with its original settings, and held-out metrics from `proxy_eval.py` and `action_diagnostic.py`.

The code lives with the ablations so the two share it. The encoders, predictors and loss are in
[models.py](../ablations/models.py) (`StackCNN`, `CNNClipEncoder`, `CausalEncoder`, `MLPPredictor(flags=True)`,
`ARPredictor(window)`), and the names are in [presets.py](../ablations/presets.py) (`HISTORY_ENCODERS`,
`HISTORY_PREDICTORS`). This folder holds the suites and plot styling ([history_presets.py](history_presets.py)), thin
wrappers around the ablation's launcher and report, and the results. These additions leave the ablations' existing
presets unchanged: the same weights, outputs and loss terms at a fixed seed (checked for all 25).

## Running it

```bash
# print every sbatch command, submit nothing
.venv/bin/python experiments/predictor_history/launch.py --name all --dry-run

# quick end-to-end check on the cluster: 10 steps per seed, evaluation included
.venv/bin/python experiments/predictor_history/launch.py --name smoke --steps 10 --anchor-steps 10

# the real run: 19 jobs (3 seeds each, in series), then a report job
.venv/bin/python experiments/predictor_history/launch.py --name all
```

- **Reuse needs matching settings.** Baselines are linked only when the existing runs used the same `--steps` and `--seeds`, so a 10-step smoke run trains them too.
- **Other suites:** `--suite option1` (or `option2`, `option3`) runs one option with its baselines.
- **Other flags:** every flag of [the ablation launcher](../ablations/launch.py) works here (`--combos`, `--constraint`, `--time`, `--no-reuse`, ...).
- **Re-plotting:** `.venv/bin/python experiments/predictor_history/report.py --name all`.

## Results

Each run goes to `results/<name>/`, with the same files as the ablations. The report in `results/<name>/report/` has:
- **`summary.md`, `summary.csv`, `per_run.csv`:** the tables.
- **`eval_comparison.png`:** every combination, in the order of the table above.
- **`training_by_arch_ref.png`:** key training curves, one column per option and encoder family, each with its baseline.
- **`top1_by_horizon_by_arch_ref.png`:** held-out top-1 at 5–25 steps ahead, the same layout.

In every figure, blue is the baseline, green the MLP predictor and orange the AR predictor. Dashed lines are the
separate-network variants (option 2) or cap 5 (option 3).
