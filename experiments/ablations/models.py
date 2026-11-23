"""Interchangeable encoders, predictors and losses for the encoder x predictor x loss ablation.

Every piece is the team's existing code (example_train.py for the reference, lewm.py for LeWM) behind two small
interfaces, so any encoder can be paired with any predictor and either loss:

    Encoder.embed(frames [B, T, 3, 64, 64]) -> embeddings in time order [B, n, D], the last one the most recent
        cnn   the reference CNN on each sliding window of HIST stacked frames: T frames -> T - HIST + 1 clip
              embeddings. context_len = 1: one clip embedding already holds the 3-frame history.
        vit   LeWM's per-frame ViT + BatchNorm projector: T frames -> T frame embeddings. context_len = HIST.

    Predictor.step(window [B, w, D], actions of those w positions) -> next embedding [B, D]
        mlp   the reference's residual MLP, over the last `window` embeddings concatenated:
              next = z_last + MLP([z_1, ..., z_w, a]). With the cnn encoder window = 1, which is exactly the
              reference predictor. With the vit encoder window = HIST ("history MLP"): same network, 3x wider
              input, so it sees the same history as the AR predictor.
        mlp1  the literal reference predictor on the last embedding only (window = 1), whatever the encoder.
              With vit it cannot see velocity; vit-mlp1 vs vit-mlp measures what the history is worth.
        ar    LeWM's causal transformer with AdaLN-zero action conditioning over up to HIST embeddings, then
              its BatchNorm projector. With vit the window starts full (o[t-10], o[t-5], o[t]) and the two
              positions before t get LeWM's learned action-mask token (the evaluator gives no past actions).
              With cnn there is one context embedding, so the window grows 1 -> 2 -> 3 with predictions and
              then slides; every position then has a real action and the mask is never used.

    The rollout is shared: start from the context, predict, append, keep the last `window` embeddings.
    Step k conditions on actions[:, :k+1] only, so z[:, k+1] is causal by construction.

Embedding size. The encoder decides D (its output is the embedding): cnn D=256 (ref_long), vit D=96 (lewm_long).
Each predictor keeps its own configuration and adapts only its input/output to D:
    mlp/mlp1  input window*D + 10, hidden 512, output D (as the reference)
    ar        keeps its 96-wide transformer; a Linear(D, 96) in front and its projector 96 -> 384 -> D behind.
              Both are identities / unchanged when D = 96, so vit-ar is exactly lewm_long.
Actions: mlp/mlp1 take them raw (as the reference); ar normalizes them with corpus statistics baked into the
model and embeds them with LeWM's Embedder (as LeWM).

Losses (both computed on the same tensors; whichever is not trained is still logged, without gradient):
    ref   25 * MSE(pred, target) + VCReg(all) + VCReg(all - clip mean) + InfoNCE(pred_k, target_k), as
          example_train.jepa_forward
    lewm  MSE(pred, target) + 0.1 * SIGReg(all), as train_lewm.lewm_forward
"all" is every embedding of the sample (6 clip embeddings for cnn, 8 frame embeddings for vit); the targets are
the HORIZON embeddings after the context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # presets.py

import example_train  # noqa: E402
import lewm  # noqa: E402
from evaluate import ACTION_DIM, HIST, HORIZON, SKIP  # noqa: E402
from stable_pretraining.backbone import MLP as SptMLP  # noqa: E402

ACTION_BLOCK = ACTION_DIM * SKIP
from presets import LOSS_PRESETS, LOSSES  # noqa: E402  (the loss presets: term weights, see presets.py)

ENCODERS, PREDICTORS = ("cnn", "vit"), ("mlp", "mlp1", "ar")

# The configurations being compared. ref_long = example_train.py defaults; lewm_long = the README's command.
REF_LONG = dict(dim=256, width=32, hidden=512)
LEWM_LONG = dict(dim=96, heads=4, dim_head=24, patch=8, enc_depth=6, enc_mlp=384, pred_depth=6, pred_mlp=384,
                 proj_hidden=384, pred_dropout=0.1, sigreg_weight=0.1)

# Optimizer settings belong to the network, not to the loss: each network keeps those of the model it comes from.
OPTIM = {
    "cnn": dict(lr=1e-3, weight_decay=1e-4, schedule="cosine", clip=None),  # example_train.py
    "mlp": dict(lr=1e-3, weight_decay=1e-4, schedule="cosine", clip=None),
    "mlp1": dict(lr=1e-3, weight_decay=1e-4, schedule="cosine", clip=None),
    "vit": dict(lr=2e-4, weight_decay=1e-3, schedule="warmup_cosine", clip=1.0),  # train_lewm.py
    "ar": dict(lr=2e-4, weight_decay=1e-3, schedule="warmup_cosine", clip=1.0),
}


# ---------------------------------------------------------------------------------------------------- encoders

class CNNEncoder(nn.Module):
    context_len = 1

    def __init__(self):
        super().__init__()
        self.dim = REF_LONG["dim"]
        self.net = example_train.Encoder(REF_LONG["dim"], REF_LONG["width"])

    def embed(self, frames):
        b, t = frames.shape[:2]
        windows = torch.stack([frames[:, k : k + HIST] for k in range(t - HIST + 1)], 1)
        return self.net(windows.flatten(0, 1)).view(b, t - HIST + 1, -1)


class ViTEncoder(nn.Module):
    context_len = HIST

    def __init__(self):
        super().__init__()
        c = LEWM_LONG
        self.dim = c["dim"]
        self.net = lewm.ViTEncoder(c["dim"], c["enc_depth"], c["heads"], c["dim_head"], c["enc_mlp"], c["patch"])
        self.projector = lewm.MLP(c["dim"], c["proj_hidden"])

    def embed(self, frames):
        b, t = frames.shape[:2]
        return self.projector(self.net(frames.flatten(0, 1))).view(b, t, -1)


# -------------------------------------------------------------------------------------------------- predictors

class MLPPredictor(nn.Module):
    """next = z_last + MLP([z_1 .. z_window, a]) on raw actions (example_train.Predictor when window = 1)."""

    def __init__(self, dim, window):
        super().__init__()
        self.window = window
        self.mlp = SptMLP(window * dim + ACTION_BLOCK, [REF_LONG["hidden"], REF_LONG["hidden"], dim],
                          activation_layer=nn.GELU)

    def embed_actions(self, actions):
        return actions

    def missing_actions(self, b, n, like):
        return like.new_zeros(b, n, ACTION_BLOCK)  # never read: the MLP only uses the newest action

    def step(self, z, a):
        if z.shape[1] < self.window:  # cannot happen with the encoders here; guard for future pairings
            z = torch.cat([z[:, :1].expand(-1, self.window - z.shape[1], -1), z], 1)
        return z[:, -1] + self.mlp(torch.cat([z.flatten(1), a[:, -1]], -1))


class ARPredictor(nn.Module):
    """LeWM's predictor: Embedder + causal AdaLN transformer + BatchNorm projector (lewm.LeWorldModel.rollout)."""

    window = HIST

    def __init__(self, dim):
        super().__init__()
        c = LEWM_LONG
        width = c["dim"]
        self.in_proj = nn.Linear(dim, width) if dim != width else nn.Identity()
        self.action_encoder = lewm.Embedder(ACTION_BLOCK, width)
        self.predictor = lewm.ARPredictor(width, c["pred_depth"], c["heads"], c["dim_head"], c["pred_mlp"],
                                          c["pred_dropout"])
        self.pred_proj = lewm.MLP(width, c["proj_hidden"], out=dim)
        self.action_mask = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.register_buffer("action_mean", torch.zeros(1, 1, ACTION_BLOCK))
        self.register_buffer("action_std", torch.ones(1, 1, ACTION_BLOCK))

    def set_action_stats(self, mean, std):
        self.action_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, 1, ACTION_BLOCK))
        self.action_std.copy_(torch.as_tensor(std, dtype=torch.float32).view(1, 1, ACTION_BLOCK))

    def embed_actions(self, actions):
        return self.action_encoder((actions - self.action_mean) / self.action_std)

    def missing_actions(self, b, n, like):
        return self.action_mask.expand(b, n, -1)

    def step(self, z, a):
        return self.pred_proj(self.predictor(self.in_proj(z), a)[:, -1])


# ------------------------------------------------------------------------------------------------- world model

class WorldModel(nn.Module):
    """The submission contract: frames [B, HIST, 3, 64, 64], actions [B, HORIZON, 2*SKIP] -> z [B, HORIZON+1, D]."""

    def __init__(self, encoder: str, predictor: str):
        super().__init__()
        self.encoder = CNNEncoder() if encoder == "cnn" else ViTEncoder()
        dim = self.encoder.dim
        if predictor == "ar":
            self.predictor = ARPredictor(dim)
        else:
            window = 1 if predictor == "mlp1" else self.encoder.context_len
            self.predictor = MLPPredictor(dim, window)

    def rollout(self, context, actions):
        """context [B, C, D] (real embeddings, newest last), actions [B, K, 2*SKIP] raw -> predictions [B, K, D].

        Action i of the sequence below is the one taken from embedding i; the C-1 context embeddings before t have
        none (the predictor supplies its stand-in)."""
        b, c, _ = context.shape
        a = self.predictor.embed_actions(actions)
        a_seq = torch.cat([self.predictor.missing_actions(b, c - 1, a), a], 1) if c > 1 else a
        z_seq = list(context.unbind(1))
        preds = []
        for _ in range(actions.shape[1]):
            n = len(z_seq)
            w = min(n, self.predictor.window)
            pred = self.predictor.step(torch.stack(z_seq[n - w :], 1), a_seq[:, n - w : n])
            z_seq.append(pred)
            preds.append(pred)
        return torch.stack(preds, 1)

    def forward(self, frames, actions):
        context = self.encoder.embed(frames)  # [B, context_len, D]
        return torch.cat([context[:, -1:], self.rollout(context, actions)], 1)


# ------------------------------------------------------------------------------------------------------ losses

def effective_rank(x):
    """exp(entropy) of the normalized covariance spectrum: ~1 when collapsed, up to D when isotropic."""
    x = x.float() - x.float().mean(0)
    eig = torch.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)).clamp_min(0)
    p = eig / eig.sum().clamp_min(1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum())


def loss_terms(model, batch, vcreg, sigreg, loss_name):
    """The weighted loss of preset `loss_name` (presets.py), plus every term and diagnostic to log.

    All five terms are computed on every step; those with weight 0 are computed without gradient, for logging only.
    The logged names keep the main ablation's (ref/infonce, ref/vcreg, ref/vcreg_t, lewm/sigreg, and the totals of
    the two original losses ref/total and lewm/total), so every run's metrics.csv has the same columns.
    Returns (loss to optimize, dict of scalars to log)."""
    w = LOSS_PRESETS[loss_name]
    frames = example_train.to_model_frames(batch["frames"])  # [B, HIST + HORIZON, 3, 64, 64]
    emb = model.encoder.embed(frames).float()  # [B, n, D], all embeddings of the sample
    c = model.encoder.context_len
    target = emb[:, c:]  # [B, HORIZON, D]
    pred = model.rollout(emb[:, :c], batch["actions"]).float()  # [B, HORIZON, D]

    builders = {
        "mse": lambda: (pred - target).square().mean(),
        "infonce": lambda: sum(example_train.info_nce(pred[:, k], target[:, k], w["tau"])
                               for k in range(HORIZON)) / HORIZON,
        "vcreg": lambda: vcreg(emb.flatten(0, 1)),
        "vcreg_t": lambda: vcreg((emb - emb.mean(1, keepdim=True)).flatten(0, 1)),
        "sigreg": lambda: sigreg(emb.transpose(0, 1)),
    }
    values = {}
    for term, build in builders.items():
        if w[term]:
            values[term] = build()
        else:
            with torch.no_grad():
                values[term] = build()
    loss = sum(w[t] * values[t] for t in builders if w[t])

    terms = {}
    with torch.no_grad():
        v = {t: x.detach() for t, x in values.items()}
        ref_nce = v["infonce"] if w["tau"] == 0.1 else sum(  # the original losses' InfoNCE uses temperature 0.1
            example_train.info_nce(pred[:, k], target[:, k]) for k in range(HORIZON)) / HORIZON
        if w["tau"] != 0.1:  # the InfoNCE this preset trains, at its own temperature
            terms["infonce_trained_tau"] = v["infonce"]
        terms.update({"ref/infonce": ref_nce, "ref/vcreg": v["vcreg"], "ref/vcreg_t": v["vcreg_t"],
                      "lewm/sigreg": v["sigreg"],
                      "ref/total": 25 * v["mse"] + v["vcreg"] + v["vcreg_t"] + ref_nce,
                      "lewm/total": v["mse"] + LEWM_LONG["sigreg_weight"] * v["sigreg"]})
        terms["loss"] = loss.detach()
        mse = v["mse"]
        terms["mse"] = mse.detach()
        per_k = (pred - target).square().mean((0, 2))
        for k in range(HORIZON):
            terms[f"mse_k{k + 1}"] = per_k[k]
        p, t = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
        terms["cos_pred_target"] = (p * t).sum(-1).mean()
        # in-batch retrieval: does each prediction pick out its own target among the batch? (what InfoNCE trains)
        hits = [(p[:, k] @ t[:, k].T).argmax(1) == torch.arange(len(p), device=p.device) for k in range(HORIZON)]
        terms["batch_top1"] = torch.stack(hits).float().mean()
        terms["emb_std"] = emb.flatten(0, 1).std(0).mean()
        terms["emb_eff_rank"] = effective_rank(emb.flatten(0, 1))
        terms["pred_std"] = pred.flatten(0, 1).std(0).mean()
    return loss, terms
