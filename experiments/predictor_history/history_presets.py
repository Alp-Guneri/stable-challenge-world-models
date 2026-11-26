"""Combinations of the predictor-history experiment: plain data, no torch.

Names are the ablation's "<encoder>-<predictor>-<loss>"; the new encoder and predictor names are defined in
experiments/ablations/presets.py (HISTORY_ENCODERS, HISTORY_PREDICTORS). Every combination uses the reference loss.
"""

# The encoder sees all 3 frames, the predictor 1 embedding: one per encoder family. Reused from the ablations.
BASELINES = ["cnn-mlp-ref", "vitst-mlp-ref"]

# Option 1, shorter encoder clips: 2-frame clips, predictor window 2 (2 real embeddings, then sliding).
OPTION1 = [f"{enc}_clip2-{pred}-ref" for enc in ("cnn", "vitst") for pred in ("mlp", "ar_w2")]

# Option 2, causal clip encoder: embeddings of [t-10], [t-10, t-5], [t-10, t-5, t]; predictor window 3.
# causal = one network for 1, 2 and 3 frames; causalsep = one network per number of frames.
OPTION2 = [f"{enc}_{mode}-{pred}-ref" for enc in ("cnn", "vitst") for mode in ("causal", "causalsep")
           for pred in ("mlp", "ar")]

# Option 3, predictor history from its own predictions: the baseline encoder (1 clip embedding); the predictor's window
# grows with its predictions up to a cap of 3 (mlp_w3, ar) or 5 (mlp_w5, ar_w5). cnn-ar-ref is reused from the
# ablations' main suite.
OPTION3 = [f"{enc}-{pred}-ref" for enc in ("cnn", "vitst") for pred in ("mlp_w3", "mlp_w5", "ar", "ar_w5")]

SUITES = {
    "all": BASELINES + OPTION1 + OPTION2 + OPTION3,
    "option1": BASELINES + OPTION1,
    "option2": BASELINES + OPTION2,
    "option3": BASELINES + OPTION3,
}
REUSE_FROM = ["main", "encoder"]  # experiments/ablations/results: the baselines, cnn-ar-ref and the blend anchors

# Report figures: one panel per option and encoder family, each with its baseline. Color = predictor (blue baseline,
# green MLP, orange AR); dashed = the variant (separate networks in option 2, cap 5 in option 3).
ARCH_STYLE = {
    "cnn-mlp": (1, "-"), "vitst-mlp": (1, "-"),
    **{f"{e}_clip2-mlp": (3, "-") for e in ("cnn", "vitst")},
    **{f"{e}_clip2-ar_w2": (2, "-") for e in ("cnn", "vitst")},
    **{f"{e}_causal-mlp": (3, "-") for e in ("cnn", "vitst")},
    **{f"{e}_causalsep-mlp": (3, "--") for e in ("cnn", "vitst")},
    **{f"{e}_causal-ar": (2, "-") for e in ("cnn", "vitst")},
    **{f"{e}_causalsep-ar": (2, "--") for e in ("cnn", "vitst")},
    **{f"{e}-mlp_w3": (3, "-") for e in ("cnn", "vitst")},
    **{f"{e}-mlp_w5": (3, "--") for e in ("cnn", "vitst")},
    **{f"{e}-ar": (2, "-") for e in ("cnn", "vitst")},
    **{f"{e}-ar_w5": (2, "--") for e in ("cnn", "vitst")},
}
ARCH_GROUPS = {
    f"{option}, {family}": [f"{enc}-mlp"] + members(enc)
    for option, members in (
        ("1: 2-frame clips", lambda e: [f"{e}_clip2-mlp", f"{e}_clip2-ar_w2"]),
        ("2: causal encoder", lambda e: [f"{e}_causal-mlp", f"{e}_causalsep-mlp", f"{e}_causal-ar",
                                         f"{e}_causalsep-ar"]),
        ("3: predictor's own history", lambda e: [f"{e}-mlp_w3", f"{e}-mlp_w5", f"{e}-ar", f"{e}-ar_w5"]),
    )
    for family, enc in (("CNN", "cnn"), ("ViT", "vitst"))
}
