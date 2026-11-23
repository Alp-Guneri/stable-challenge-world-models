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
ENCODERS = ("cnn", *ENCODER_VARIANTS)
PREDICTORS = ("mlp", "mlp1", "ar")


def vit_config(name):
    return {**VIT_BASE, **ENCODER_VARIANTS[name]}


# ---------------------------------------------------------------------------------------------------- suites

SUITES = {
    # the encoder x predictor x loss grid, plus the literal reference MLP on per-frame embeddings
    "main": [f"{e}-{p}-{l}" for e in ("cnn", "vit") for p in ("mlp", "ar") for l in ("ref", "lewm")]
            + ["vit-mlp1-ref"],
    # the best architecture (CNN encoder, MLP predictor) with every loss preset
    "loss": [f"cnn-mlp-{l}" for l in LOSSES],
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
