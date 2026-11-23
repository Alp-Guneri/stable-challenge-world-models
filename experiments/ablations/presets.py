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

SUITES = {
    # the encoder x predictor x loss grid, plus the literal reference MLP on per-frame embeddings
    "main": [f"{e}-{p}-{l}" for e in ("cnn", "vit") for p in ("mlp", "ar") for l in ("ref", "lewm")]
            + ["vit-mlp1-ref"],
    # the best architecture (CNN encoder, MLP predictor) with every loss preset
    "loss": [f"cnn-mlp-{l}" for l in LOSSES],
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
