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

Predictor-history variants (presets.HISTORY_ENCODERS / HISTORY_PREDICTORS, experiments/predictor_history/README.md):
    <base>_clip2       CNNClipEncoder / ClipViT(frames=2): 2-frame clips, context of 2 embeddings
    <base>_causal      CausalEncoder: [t-10], [t-10, t-5], [t-10, t-5, t] from one network (context of 3; the first
    <base>_causalsep   2 are "partial", left out of the regularizers); causalsep: one network per number of frames
    mlp_w3, mlp_w5     MLPPredictor(flags=True): fixed window of 3 / 5 slots, empty slots zero-filled and flagged
    ar_w2, ar_w5       ARPredictor(window=2 / 5)
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
from presets import (ENCODER_VARIANTS, ENCODERS, HISTORY_ENCODERS, HISTORY_PREDICTORS, LOSS_PRESETS,  # noqa: E402,F401
                     LOSSES, PREDICTORS, encoder_config, vit_config)

# The configurations being compared. ref_long = example_train.py defaults; lewm_long = the README's command.
REF_LONG = dict(dim=256, width=32, hidden=512)
LEWM_LONG = dict(dim=96, heads=4, dim_head=24, patch=8, enc_depth=6, enc_mlp=384, pred_depth=6, pred_mlp=384,
                 proj_hidden=384, pred_dropout=0.1, sigreg_weight=0.1)

# Optimizer settings belong to the network, not to the loss: each network keeps those of the model it comes from.
REF_OPTIM = dict(lr=1e-3, weight_decay=1e-4, schedule="cosine", clip=None)  # example_train.py
LEWM_OPTIM = dict(lr=2e-4, weight_decay=1e-3, schedule="warmup_cosine", clip=1.0)  # train_lewm.py
OPTIM = {
    "cnn": REF_OPTIM, "mlp": REF_OPTIM, "mlp1": REF_OPTIM, "ar": LEWM_OPTIM,
    # every ViT variant keeps LeWM's settings, except the learning rate when that is the variant's one change
    **{name: {**LEWM_OPTIM, "lr": vit_config(name)["lr"]} for name in ENCODER_VARIANTS},
}
# predictor-history variants keep the settings of the network they are built from
OPTIM.update({name: OPTIM[base] for name, (base, _) in HISTORY_ENCODERS.items()})
OPTIM.update({name: OPTIM[family] for name, (family, _) in HISTORY_PREDICTORS.items()})


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


# The ViT family (presets.ENCODER_VARIANTS). "vit" is exactly lewm_long's encoder: D = 96, 6 layers, 4 heads of 24,
# transformer MLP 384, 8x8 patches, BatchNorm projector 96 -> 384 -> 96 (same structure and parameter count as the
# main suite's ViT). D sets every width: head size D/4, transformer MLP and projector hidden 4D.
#
# Attention: these encoders use PyTorch's scaled_dot_product_attention, the same computation as lewm.Attention's
# explicit softmax(QK^T/sqrt(d))V (checked numerically equal), but without materializing the attention matrix: with
# 4x4 patches (257 tokens per frame) the explicit matrices would need ~1 GB per layer per batch, more than the 8 GB
# GPUs have. It exports to ONNX at opset 17. Encoder attention is never causal: all its frames are at or before t.

class SDPAttention(lewm.Attention):
    """lewm.Attention with the same parameters, computed by scaled_dot_product_attention."""

    def forward(self, x):
        b, t, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=self.causal,
                                             dropout_p=self.drop.p if self.training else 0.0)
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


def vit_backbone(cfg, in_chans=3):
    """lewm.ViTEncoder with cfg's width and patch size (and in_chans input channels), on SDPA attention."""
    d = cfg["dim"]
    net = lewm.ViTEncoder(d, cfg["depth"], cfg["heads"], d // cfg["heads"], 4 * d, cfg["patch"])
    if in_chans != 3:
        net.patch_embed = nn.Conv2d(in_chans, d, cfg["patch"], stride=cfg["patch"])
    for m in net.modules():
        if type(m) is lewm.Attention:
            m.__class__ = SDPAttention
    return net


def projector(cfg):
    """LeWM's projector MLP (D -> 4D -> D), with BatchNorm (LeWM) or LayerNorm in the hidden layer."""
    d = cfg["dim"]
    if cfg["proj_norm"] == "batch":
        return lewm.MLP(d, 4 * d)
    return nn.Sequential(nn.Linear(d, 4 * d), nn.LayerNorm(4 * d), nn.GELU(), nn.Linear(4 * d, d))


class FrameViT(nn.Module):
    """input = frame: LeWM's encoder, one embedding per frame; the context is the 3 frame embeddings."""

    context_len = HIST

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg["dim"]
        self.net = vit_backbone(cfg)
        self.projector = projector(cfg)

    def embed(self, frames):
        b, t = frames.shape[:2]
        return self.projector(self.net(frames.flatten(0, 1))).view(b, t, -1)


class StackViT(nn.Module):
    """input = stack: the clip's frames (HIST unless `frames` says otherwise) as one 3*frames-channel image -> 64
    patch tokens (each covers its square in all frames) + CLS -> one embedding of the clip."""

    def __init__(self, cfg, frames=HIST):
        super().__init__()
        self.vit = vit_backbone(cfg, in_chans=3 * frames)

    def forward(self, clips):
        """clips [N, frames, 3, S, S] in [0, 1] -> [N, D]."""
        v = self.vit
        x = ((clips - v.mean.unsqueeze(1)) / v.std.unsqueeze(1)).flatten(1, 2)  # per-frame ImageNet normalization
        x = v.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), x], 1) + v.pos_embedding
        for block in v.blocks:
            x = block(x)
        return v.norm(x)[:, 0]


class SpaceTimeViT(nn.Module):
    """input = spacetime: one token per patch per frame (frames x 64), each with the patch's spatial embedding plus
    its frame's temporal embedding, + CLS; attention across space and time -> one embedding of the clip.

    `frames` (default HIST) is the most frames a clip can have. A shorter clip just has fewer tokens: its frames take
    the last time embeddings, so the newest frame always has the same one (a clip stands for its newest frame)."""

    def __init__(self, cfg, frames=HIST):
        super().__init__()
        self.vit = vit_backbone(cfg)  # patch embedding shared by the frames; its pos_embedding is the spatial one
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, 1, cfg["dim"]))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(self, clips):
        """clips [N, f, 3, S, S] in [0, 1], f <= frames -> [N, D]."""
        v, (n, f) = self.vit, clips.shape[:2]
        x = (clips.flatten(0, 1) - v.mean) / v.std
        x = v.patch_embed(x).flatten(2).transpose(1, 2)  # [N*f, P, D]
        x = x.view(n, f, -1, x.shape[-1]) + v.pos_embedding[:, 1:].unsqueeze(1) + self.time_embedding[:, -f:]
        cls = v.cls_token.expand(n, -1, -1) + v.pos_embedding[:, :1]
        x = torch.cat([cls, x.flatten(1, 2)], 1)  # [N, 1 + f*P, D]
        for block in v.blocks:
            x = block(x)
        return v.norm(x)[:, 0]


class ClipViT(nn.Module):
    """input = stack or spacetime: one embedding per sliding window of `frames` frames, like the CNN. With the default
    HIST frames the context is 1 clip embedding; with 2 (clip2) it is 2: (t-10, t-5) and (t-5, t)."""

    def __init__(self, cfg, frames=HIST):
        super().__init__()
        self.dim, self.frames, self.context_len = cfg["dim"], frames, HIST - frames + 1
        self.net = StackViT(cfg, frames) if cfg["input"] == "stack" else SpaceTimeViT(cfg, frames)
        self.projector = projector(cfg)

    def embed(self, frames):
        b, t, c = *frames.shape[:2], self.frames
        windows = torch.stack([frames[:, k : k + c] for k in range(t - c + 1)], 1)
        return self.projector(self.net(windows.flatten(0, 1))).view(b, t - c + 1, -1)


# Predictor-history encoders (presets.HISTORY_ENCODERS, experiments/predictor_history).

class StackCNN(nn.Module):
    """The reference CNN (example_train.Encoder's layers, widths and D) over `frames` stacked frames: 3*frames input
    channels, plus with `flags` one constant 0/1 "frame present" channel per frame slot. A clip with fewer frames
    fills the oldest slots: zeros (after the reference's centering, so mid-grey) with their flag at 0."""

    def __init__(self, frames, flags=False):
        super().__init__()
        self.frames, self.flags = frames, flags
        self.net = example_train.Encoder(REF_LONG["dim"], REF_LONG["width"])
        chans = 3 * frames + (frames if flags else 0)
        if chans != 3 * HIST:
            self.net.conv[0] = nn.Conv2d(chans, REF_LONG["width"], 4, stride=2, padding=1)

    def forward(self, clips):
        """clips [N, f, 3, S, S] in [0, 1], f <= frames (f < frames needs flags) -> [N, D]."""
        # Padding and flags are computed from the input (Pad op, x * 0 + flag), not built as batch-sized constants:
        # ONNX export would store those in the file (~9 MB of zeros and ones for the causal CNN).
        f = clips.shape[1]
        x = clips - 0.5  # as example_train.Encoder
        if f < self.frames:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, self.frames - f, 0))  # zero frames in the oldest slots
        if self.flags:
            present = x.new_tensor([0.0] * (self.frames - f) + [1.0] * f).view(1, -1, 1, 1, 1)
            return self.net.proj(self.net.conv(torch.cat([x.flatten(1, 2), (x[:, :, :1] * 0 + present).flatten(1, 2)],
                                                         1)).flatten(1))
        return self.net.proj(self.net.conv(x.flatten(1, 2)).flatten(1))


class CNNClipEncoder(nn.Module):
    """clip2: the reference CNN on every sliding window of `frames` stacked frames (3 frames -> HIST - frames + 1
    context embeddings)."""

    def __init__(self, frames):
        super().__init__()
        self.dim, self.frames, self.context_len = REF_LONG["dim"], frames, HIST - frames + 1
        self.net = StackCNN(frames)

    def embed(self, frames):
        b, t, c = *frames.shape[:2], self.frames
        windows = torch.stack([frames[:, k : k + c] for k in range(t - c + 1)], 1)
        return self.net(windows.flatten(0, 1)).view(b, t - c + 1, -1)


class CausalEncoder(nn.Module):
    """causal / causalsep: embedding i sees the frames up to i within the last HIST. A sequence f0 .. f(T-1) gives T
    embeddings: [f0], [f0, f1] (the HIST - 1 partial ones, context only), then every full window [f(k-2), f(k-1), f(k)]
    (the last context embedding = z[:, 0], and every target). The forward pass's 3 frames give the 3 context ones.

    shared (causal): one network for 1, 2 or 3 frames. CNN: StackCNN with presence flags (missing frames zero-filled).
        ViT: SpaceTimeViT, missing frames' tokens left out, time embeddings aligned to the newest frame; one projector,
        applied to partial and full embeddings in one call so its BatchNorm statistics (batch in training, running in
        evaluation) are of the same mix.
    separate (causalsep): one network (and, for the ViT, one projector) per number of frames; the 3-frame one is the
        baseline encoder (cnn or vitst) itself.
    `n_partial` tells loss_terms which embeddings are partial: they are left out of the regularizers (VCReg / SIGReg)
    and logged separately."""

    context_len = HIST
    n_partial = HIST - 1

    def __init__(self, base, separate):
        super().__init__()
        self.separate = separate
        sizes = range(1, HIST + 1) if separate else [HIST]
        if base == "cnn":
            self.dim = REF_LONG["dim"]
            self.nets = nn.ModuleList(StackCNN(n, flags=not separate) for n in sizes)
            self.projectors = nn.ModuleList(nn.Identity() for _ in sizes)
        else:
            cfg = vit_config(base)
            assert cfg["input"] == "spacetime", f"causal encoders need a spacetime ViT, not {base}"
            self.dim = cfg["dim"]
            self.nets = nn.ModuleList(SpaceTimeViT(cfg, n) for n in sizes)
            self.projectors = nn.ModuleList(projector(cfg) for _ in sizes)

    def embed(self, frames):
        b, t = frames.shape[:2]
        groups = [frames[:, :n] for n in range(1, HIST)]  # partial: [B, n, 3, S, S]
        groups.append(torch.stack([frames[:, k : k + HIST] for k in range(t - HIST + 1)], 1).flatten(0, 1))
        if self.separate:
            out = [self.projectors[g.shape[1] - 1](self.nets[g.shape[1] - 1](g)) for g in groups]
        else:
            feats = [self.nets[0](g) for g in groups]
            out = self.projectors[0](torch.cat(feats)).split([len(f) for f in feats])
        return torch.cat([torch.stack(out[:-1], 1), out[-1].view(b, t - HIST + 1, -1)], 1)


def make_encoder(name):
    if name in HISTORY_ENCODERS:
        base, mode = HISTORY_ENCODERS[name]
        if mode == "clip2":
            return CNNClipEncoder(2) if base == "cnn" else ClipViT(vit_config(base), frames=2)
        return CausalEncoder(base, separate=mode == "causalsep")
    if name == "cnn":
        return CNNEncoder()
    cfg = vit_config(name)
    return FrameViT(cfg) if cfg["input"] == "frame" else ClipViT(cfg)


# -------------------------------------------------------------------------------------------------- predictors

class MLPPredictor(nn.Module):
    """next = z_last + MLP([z_1 .. z_window, a]) on raw actions (example_train.Predictor when window = 1).

    With `flags` (mlp_w3, mlp_w5) the window may be only partly filled (it grows with the predictions): the empty
    (oldest) slots are zeros and one 0/1 "filled" input per slot, before the action, says which slots are real."""

    def __init__(self, dim, window, flags=False):
        super().__init__()
        self.window, self.flags = window, flags
        self.mlp = SptMLP(window * dim + (window if flags else 0) + ACTION_BLOCK,
                          [REF_LONG["hidden"], REF_LONG["hidden"], dim], activation_layer=nn.GELU)

    def embed_actions(self, actions):
        return actions

    def missing_actions(self, b, n, like):
        return like.new_zeros(b, n, ACTION_BLOCK)  # never read: the MLP only uses the newest action

    def step(self, z, a):
        b, n = z.shape[:2]
        if self.flags:
            filled = z[:, -1, :1] * 0 + z.new_tensor([0.0] * (self.window - n) + [1.0] * n)  # [B, window]
            slots = F.pad(z, (0, 0, self.window - n, 0)) if n < self.window else z  # zeros in the oldest slots
            return z[:, -1] + self.mlp(torch.cat([slots.flatten(1), filled, a[:, -1]], -1))
        if n < self.window:  # cannot happen with the pairings run (flags handle growing windows); guard only
            z = torch.cat([z[:, :1].expand(-1, self.window - n, -1), z], 1)
        return z[:, -1] + self.mlp(torch.cat([z.flatten(1), a[:, -1]], -1))


class ARPredictor(nn.Module):
    """LeWM's predictor: Embedder + causal AdaLN transformer + BatchNorm projector (lewm.LeWorldModel.rollout).
    `window` (default HIST) caps the embeddings it sees, and sets its number of position embeddings."""

    def __init__(self, dim, window=HIST):
        super().__init__()
        c = LEWM_LONG
        width = c["dim"]
        self.window = window
        self.in_proj = nn.Linear(dim, width) if dim != width else nn.Identity()
        self.action_encoder = lewm.Embedder(ACTION_BLOCK, width)
        self.predictor = lewm.ARPredictor(width, c["pred_depth"], c["heads"], c["dim_head"], c["pred_mlp"],
                                          c["pred_dropout"], length=window)
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
        self.encoder = make_encoder(encoder)
        dim = self.encoder.dim
        if predictor in HISTORY_PREDICTORS:
            family, window = HISTORY_PREDICTORS[predictor]
            self.predictor = ARPredictor(dim, window) if family == "ar" else MLPPredictor(dim, window, flags=True)
        elif predictor == "ar":
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
    # The regularizers see the full embeddings only: a causal encoder's first n_partial embeddings (context only,
    # from fewer frames) are left out, so the regularized set is the same as the baseline's. 0 for every other encoder.
    n_partial = getattr(model.encoder, "n_partial", 0)
    full = emb[:, n_partial:]

    builders = {
        "mse": lambda: (pred - target).square().mean(),
        "infonce": lambda: sum(example_train.info_nce(pred[:, k], target[:, k], w["tau"])
                               for k in range(HORIZON)) / HORIZON,
        "vcreg": lambda: vcreg(full.flatten(0, 1)),
        "vcreg_t": lambda: vcreg((full - full.mean(1, keepdim=True)).flatten(0, 1)),
        "sigreg": lambda: sigreg(full.transpose(0, 1)),
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
        terms["emb_std"] = full.flatten(0, 1).std(0).mean()
        terms["emb_eff_rank"] = effective_rank(full.flatten(0, 1))
        if n_partial:  # are the partial embeddings informative, or ignored by the predictor?
            partial = emb[:, :n_partial].flatten(0, 1)
            terms["emb_partial_std"] = partial.std(0).mean()
            terms["emb_partial_eff_rank"] = effective_rank(partial)
        terms["pred_std"] = pred.flatten(0, 1).std(0).mean()
    return loss, terms
