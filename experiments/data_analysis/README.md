# Data analysis behind slides 3 and 4

What the training data looks like before any model is trained: where the dot goes in each environment, and
how much of its movement a simple straight-line rule can predict from the model's own inputs. This is a data
baseline. **It is not a model result.**

The data has no environment labels, so environments are named by clip length:

- **Environment 1** = the 100-frame clips (green dot in a U-shaped corridor), 1,700 clips
- **Environment 2** = the 50-frame clips (red dot in two rooms joined by a door), 1,632 clips

Clip length matches what the first frame looks like in all 3,332 clips.

## Run it

From the repo root, with `train.npz` in `corpus/` as in CHALLENGE.md. CPU is enough; no GPU or Slurm needed.

```bash
.venv/bin/python experiments/data_analysis/data_findings.py       # ~3 min: every number   -> results/stats.json
.venv/bin/python experiments/data_analysis/checks.py              # ~2 min: extra checks   -> results/checks.json
.venv/bin/python experiments/data_analysis/plot_data_findings.py  # ~1 min: full figures   -> results/env1.png, env2.png, tracking_check.png
```

(`.venv/bin/python` rather than `uv run`, as in experiments/ablations: `uv run` re-syncs the venv to uv.lock.)

## What each file does

| File | What it does |
|---|---|
| `data_findings.py` | Finds the dot in every frame, builds the heat maps, and fits and scores the straight-line rules. Writes `stats.json`, which holds every number on the slides. |
| `checks.py` | Extra checks on the analysis: the action timing check (lag), errors near walls in pixels, holes in the map, the dot never being inside the wall, scene identity, and the trainer's draw share. Writes `checks.json`. |
| `plot_data_findings.py` | One full figure per environment (map + chart), plus `tracking_check.png`, which shows random frames with the detected dot centre. |

## How the numbers are made

1. **Which clips.** Only the first 2,999 clips are used (`corpus_loader.episode_split(c, 0.1)`). The last 333 are
   held out for `proxy_eval.py` and are never analysed. Clip counts, and the length-vs-look check, use all clips.
2. **Where the dot is.** In each frame, the dot is the centre of its coloured pixels: green in Environment 1, red
   in Environment 2 (`dot_mask`, `track`). It is found in 100% and 99.96% of frames. The data stores no positions.
3. **The map (left of each slide).** Count how many frames the dot's centre spends on each pixel; darker = more
   frames. The colour scale is linear, and the busiest 5% of pixels share the darkest shade. The caption
   compares time with space: the share of frames in the 2-pixel band along the edge of the area the dot
   reaches, against the share of the area that band covers.
4. **The chart (right of each slide).**
   - **Moments.** The same windows as `example_train.py`: every frame t with 10 frames before it and 25 after it.
   - **What is predicted.** How far the dot moves from frame t to frame t+N, as x and y, for N = 1, 5, 10, 15, 20, 25.
   - **The inputs, one per line.** Green: recent motion, the two moves between the 3 input frames (t−10 → t−5 → t).
     Orange: the N actions in between (`a[t]`, …, `a[t+N−1]`). Blue: both at once, as one rule.
   - **The rule.** Least squares with a constant (`fit_predict`): prediction = weights × inputs + constant.
   - **The score.** R² on clips the rule never saw (`r2`): R² = 1 − (rule's squared error) / (squared error of always
     guessing the average movement), with x and y pooled. 1 = perfect, 0 = no better than guessing the average.
   - **Averaging.** Fit on 80% of the clips, score on the other 20%, repeated for 5 random splits (seeds 0 to 4);
     each point on the chart is the mean. The 5 splits differ by at most 0.03 (Environment 1) and 0.07 (Environment 2).
5. **Action correlation (slide header).** Each action is paired with the next one in the same clip, and the
   correlation is computed for x and y separately, then averaged: 0.00 in Environment 1, 0.80 in Environment 2.

## Where each slide number lives in `stats.json`

| On the slide | Key |
|---|---|
| clips, frames per clip | `envK.clips`, `envK.frames_per_clip` |
| action range | `envK.action_min`, `envK.action_max` |
| correlation between consecutive actions | `envK.action_autocorr_by_lag["1"]` (mean of x and y) |
| "Within 2 px of a wall or edge in …" | `envK.time_within_2px_of_edge` |
| line values | `envK.predictability_heldout_r2[N].both / actions_only / recent_motion_only` |
| analysed and held-out clips | `envK.analysis_clips`, `envK.heldout_clips` |

## What this does not show

- **Why** the dot moves this way. These are correlations from a straight-line rule, not physics.
- **How our models score.** A straight-line rule on the dot's measured position is not a world model.
- **How accurate the tracking is.** The data has no true positions to compare against.
