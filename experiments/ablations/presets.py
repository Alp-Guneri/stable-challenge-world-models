"""Loss presets and suites of combinations: plain data, no torch, so the launcher and the report can import it fast.

A combination is "<encoder>-<predictor>-<loss>". Names are split on "-", so loss presets use "_" instead.

Every loss is the same weighted sum; a preset only sets the weights (and the InfoNCE temperature):
    loss = mse * MSE(pred, target) + infonce * InfoNCE_tau(pred_k, target_k) + vcreg * VCReg(all)
           + vcreg_t * VCReg(all - clip mean) + sigreg * SIGReg(all)
All five terms are logged on every step whatever their weight, so every run has the same curves.
"""

LOSS_PRESETS = {
    # the two original losses (identical to the main ablation's)
    "ref": dict(mse=25, infonce=1, tau=0.1, vcreg=1, vcreg_t=1, sigreg=0),  # example_train.jepa_forward
    "lewm": dict(mse=1, infonce=0, tau=0.1, vcreg=0, vcreg_t=0, sigreg=0.1),  # train_lewm.lewm_forward
    # which term of the reference loss matters?
    "ref_nonce": dict(mse=25, infonce=0, tau=0.1, vcreg=1, vcreg_t=1, sigreg=0),
    "ref_novcregt": dict(mse=25, infonce=1, tau=0.1, vcreg=1, vcreg_t=0, sigreg=0),
    "lewm_nce": dict(mse=1, infonce=1, tau=0.1, vcreg=0, vcreg_t=0, sigreg=0.1),
    # is InfoNCE under-tuned?
    "ref_nce2x": dict(mse=25, infonce=2, tau=0.1, vcreg=1, vcreg_t=1, sigreg=0),
    "ref_nce05x": dict(mse=25, infonce=0.5, tau=0.1, vcreg=1, vcreg_t=1, sigreg=0),
    "ref_tau005": dict(mse=25, infonce=1, tau=0.05, vcreg=1, vcreg_t=1, sigreg=0),
    "ref_tau02": dict(mse=25, infonce=1, tau=0.2, vcreg=1, vcreg_t=1, sigreg=0),
}
ABLATION_LOSSES = tuple(LOSS_PRESETS)  # the loss suite's presets

# InfoNCE experiments (experiments/infonce): further loss presets, registered here so that train.py and the launcher
# accept them. Their suites, report panels and colors are in experiments/infonce/infonce_presets.py.
# Weight grid: every InfoNCE weight in NCE_GRID x MSE weight in MSE_GRID, the rest as ref. ref_nce2x and ref_nce05x
# above are the grid's (2, 25) and (0.5, 25) points, so they are not repeated.
NCE_GRID, MSE_GRID = (0.5, 2, 4, 8, 16, 32), (25, 12, 6)


def grid_name(nce, mse):
    """ref_nce<w>x at ref's MSE weight (as ref_nce2x), ref_nce<w>x_mse<m> otherwise. A decimal weight drops its point,
    as the loss suite's ref_nce05x (InfoNCE 0.5)."""
    return f"ref_nce{nce:g}x".replace(".", "") + ("" if mse == 25 else f"_mse{mse}")


INFONCE_PRESETS = {grid_name(n, m): dict(mse=m, infonce=n, tau=0.1, vcreg=1, vcreg_t=1, sigreg=0)
                   for m in MSE_GRID for n in NCE_GRID if grid_name(n, m) not in LOSS_PRESETS}
LOSS_PRESETS.update(INFONCE_PRESETS)
LOSSES = tuple(LOSS_PRESETS)

# How the report groups losses into panels (at most 5 lines each), and each loss's color slot and line style. Color
# follows the loss across every figure; the two settings of one knob share a color (solid = up, dashed = down).
LOSS_GROUPS = {
    "Which term matters": ["ref", "ref_nonce", "ref_novcregt", "lewm", "lewm_nce"],
    "InfoNCE weight and temperature": ["ref", "ref_nce2x", "ref_nce05x", "ref_tau005", "ref_tau02"],
}
LOSS_STYLE = {  # (categorical slot 1-8, line style)
    "ref": (1, "-"), "lewm": (2, "-"), "ref_nonce": (3, "-"), "ref_novcregt": (4, "-"), "lewm_nce": (5, "-"),
    "ref_nce2x": (6, "-"), "ref_nce05x": (6, "--"), "ref_tau005": (7, "--"), "ref_tau02": (7, "-"),
}

# ------------------------------------------------------------------------------------------------- encoders

# The ViT family. "vit" is LeWM's encoder with lewm_long's settings; every other variant changes one thing.
#   input     frame      one embedding per frame (context of 3 frame embeddings), as LeWM
#             stack      the 3 frames stacked as 9 channels before the patches: one clip embedding (as the CNN)
#             spacetime  one token per patch per frame (+ temporal embeddings), joint attention: one clip embedding
#   dim       embedding width D; heads stay 4, so head size, transformer MLP and projector width all scale with D
#   patch     patch size (64x64 frames: 8 -> 64 tokens per frame, 4 -> 256, 16 -> 16)
#   proj_norm normalization inside the projector MLP (LeWM: BatchNorm)
#   lr        the encoder's learning rate (the rest of its optimizer stays LeWM's)
VIT_BASE = dict(input="frame", dim=96, heads=4, depth=6, patch=8, proj_norm="batch", lr=2e-4)
ENCODER_VARIANTS = {
    "vit": {},
    "vitstack": dict(input="stack"),
    "vitst": dict(input="spacetime"),
    "vit_lr1e3": dict(lr=1e-3),
    "vit_lr5e5": dict(lr=5e-5),
    "vit_p4": dict(patch=4),
    "vit_p16": dict(patch=16),
    "vit_d192": dict(dim=192),
    "vit_d48": dict(dim=48),
    "vit_ln": dict(proj_norm="layer"),
}
# Predictor-history variants (experiments/predictor_history): how the history is split between encoder and predictor.
# Encoders "<base>_<mode>", base = cnn or vitst (the spacetime ViT):
#   clip2      each embedding sees 2 consecutive frames: context (t-10, t-5), (t-5, t), 2 embeddings
#   causal     each embedding sees the frames up to its own within the last 3: context [t-10], [t-10, t-5],
#              [t-10, t-5, t], 3 embeddings; one network for 1, 2 or 3 frames (CNN: missing frames zero-filled plus
#              one "frame present" channel per slot; ViT: missing frames' tokens left out)
#   causalsep  the same with a separate network (and projector) for each number of frames
HISTORY_ENCODERS = {f"{base}_{mode}": (base, mode) for base in ("cnn", "vitst")
                    for mode in ("clip2", "causal", "causalsep")}
# Predictors "<family>_w<N>": the window (embeddings the predictor sees) is capped at N; with fewer context embeddings
# it fills with the predictor's own predictions as the rollout goes ("mlp" and "ar" keep their windows: the encoder's
# context length, and 3).
#   mlp_w3, mlp_w5   the reference MLP over N slots; empty slots are zero-filled, plus one 0/1 "filled" input per slot
#   ar_w2, ar_w5     LeWM's AR predictor with N positions
HISTORY_PREDICTORS = {"mlp_w3": ("mlp", 3), "mlp_w5": ("mlp", 5), "ar_w2": ("ar", 2), "ar_w5": ("ar", 5)}

ENCODERS = ("cnn", *ENCODER_VARIANTS, *HISTORY_ENCODERS)
PREDICTORS = ("mlp", "mlp1", "ar", *HISTORY_PREDICTORS)


def vit_config(name):
    if name in HISTORY_ENCODERS:
        return vit_config(HISTORY_ENCODERS[name][0])
    return {**VIT_BASE, **ENCODER_VARIANTS[name]}


def encoder_config(name):
    """Settings of encoder `name` for config.json: None for the plain CNN."""
    if name in HISTORY_ENCODERS:
        base, mode = HISTORY_ENCODERS[name]
        return {"base": base, "history": mode, **(vit_config(base) if base != "cnn" else {})}
    return None if name == "cnn" else vit_config(name)


# ---------------------------------------------------------------------------------------------------- suites

SUITES = {
    # the encoder x predictor x loss grid, plus the literal reference MLP on per-frame embeddings
    "main": [f"{e}-{p}-{l}" for e in ("cnn", "vit") for p in ("mlp", "ar") for l in ("ref", "lewm")]
            + ["vit-mlp1-ref"],
    # the best architecture (CNN encoder, MLP predictor) with every loss preset
    "loss": [f"cnn-mlp-{l}" for l in ABLATION_LOSSES],
    # why is the ViT behind the CNN? every ViT variant with the MLP predictor and the reference loss. Controls
    # (reused from main with --reuse-from main): cnn-mlp (the target), vit-mlp (the baseline), vit-mlp1 (no history)
    "encoder": ["cnn-mlp-ref", "vit-mlp-ref", "vit-mlp1-ref", "vitstack-mlp-ref", "vitst-mlp-ref",
                "vit_lr1e3-mlp-ref", "vit_lr5e5-mlp-ref", "vit_p4-mlp-ref", "vit_p16-mlp-ref",
                "vit_d192-mlp-ref", "vit_d48-mlp-ref", "vit_ln-mlp-ref"],
}

# How the report draws encoder-predictor pairs: (categorical slot 1-8, line style). The main suite's five pairs keep
# their colors; the encoder suite's variants use the remaining slots, and the two settings of one knob share a color
# (solid = larger value, dashed = smaller). Slots 2 and 4 are reused (cnn-ar/vitstack, vit-ar/vitst): those pairs
# never run in the same suite.
ARCH_STYLE = {
    "cnn-mlp": (1, "-"), "cnn-ar": (2, "-"), "vit-mlp": (3, "-"), "vit-ar": (4, "-"), "vit-mlp1": (3, "--"),
    "vitstack-mlp": (2, "-"), "vitst-mlp": (4, "-"), "vit_ln-mlp": (5, "-"),
    "vit_lr1e3-mlp": (6, "-"), "vit_lr5e5-mlp": (6, "--"),
    "vit_p16-mlp": (7, "-"), "vit_p4-mlp": (7, "--"),
    "vit_d192-mlp": (8, "-"), "vit_d48-mlp": (8, "--"),
}
MAIN_ARCHS = ["cnn-mlp", "cnn-ar", "vit-mlp", "vit-ar", "vit-mlp1"]  # the main suite's pairs, in its figures' order
# Panels of the encoder suite's figures: each compares variants with the ViT baseline and the CNN target.
ARCH_GROUPS = {
    "How the frames enter": ["cnn-mlp", "vit-mlp", "vit-mlp1", "vitstack-mlp", "vitst-mlp"],
    "Learning rate": ["cnn-mlp", "vit-mlp", "vit_lr5e5-mlp", "vit_lr1e3-mlp"],
    "Patch size": ["cnn-mlp", "vit-mlp", "vit_p4-mlp", "vit_p16-mlp"],
    "Embedding size": ["cnn-mlp", "vit-mlp", "vit_d48-mlp", "vit_d192-mlp"],
    "Projector normalization": ["cnn-mlp", "vit-mlp", "vit_ln-mlp"],
}

# Term of each logged loss column, to label the ones a preset does not train ("monitored, not trained").
COLUMN_TERM = {"train/mse": "mse", **{f"train/mse_k{k}": "mse" for k in range(1, 6)},
               "train/ref/infonce": "infonce", "train/ref/vcreg": "vcreg", "train/ref/vcreg_t": "vcreg_t",
               "train/lewm/sigreg": "sigreg"}


def trains(loss, column):
    """False if `column` is a loss term (or one of the two original totals) that preset `loss` gives no weight."""
    if column in ("train/ref/total", "train/lewm/total"):
        return column.split("/")[1] == loss
    term = COLUMN_TERM.get(column)
    return term is None or loss not in LOSS_PRESETS or LOSS_PRESETS[loss][term] != 0
