"""LeWorldModel (LeWM) for the challenge contract.

Architecture and objective follow Maes, Le Lidec, Scieur, LeCun & Balestriero, "LeWorldModel:
Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (arXiv:2603.19312) and
the authors' reference code at github.com/lucas-maes/le-wm:

    encoder    per-frame ViT; the [CLS] token goes through a 1-layer BatchNorm MLP projector
               (the paper's reason for the projector: the ViT's final LayerNorm would otherwise
               keep SIGReg from being optimised) -> z
    predictor  causal transformer over the last HIST embeddings; the action of each step is
               injected with zero-initialised AdaLN, then the same kind of MLP projector
    objective  next-embedding MSE + lambda * SIGReg, and nothing else: no EMA, no teacher,
               no stop-gradient (see train_lewm.py)

SIGReg, the AdaLN-zero blocks, the action embedder and the projector MLP are the authors' code,
adapted only where the challenge's inputs differ (64x64 frames, 20 MiB ONNX cap).

Two things the contract forces that the paper's setting does not:

  * Sizes are scaled down. The paper's model is ~15M parameters (ViT-Tiny/patch-14 at 224px, a
    6-layer 16-head predictor with 2048-wide MLPs), which is ~60 MB of float32 weights, three
    times the 20 MiB submission cap. Depth and width are reduced to fit, with the encoder
    getting the larger share of the budget.
  * Past actions are masked. The evaluator supplies the actions from t onwards only, so the
    predictor positions holding o[t-10] and o[t-5] have no action to condition on; they get a
    learned mask embedding instead, in training and at inference alike.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import ACTION_DIM, HIST, HORIZON, SIZE, SKIP

ACTION_BLOCK = ACTION_DIM * SKIP  # the actions between two model steps, flattened
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SIGReg(nn.Module):
    """Sketched-Isotropic-Gaussian Regularizer: sliced Epps-Pulley normality test.

    The authors' single-GPU implementation. Embeddings are projected on ``num_proj`` random
    unit directions and the univariate Epps-Pulley statistic of each projection is averaged;
    by Cramer-Wold, driving it to zero matches the whole distribution to N(0, I).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: [T, B, D] -> scalar, the statistic averaged over directions and time."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head attention, optionally causal, written out so the ONNX export is plain matmuls."""

    def __init__(self, dim, heads, dim_head, dropout=0.0, causal=False):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale, self.causal = heads, dim_head**-0.5, causal
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
            logits = logits.masked_fill(mask, float("-inf"))
        out = self.drop(logits.softmax(-1)) @ v
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Standard pre-norm transformer block (the encoder)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning (the predictor)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout, causal=True)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, -1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class MLP(nn.Module):
    """The paper's projector: one hidden layer with BatchNorm, which SIGReg needs in place of
    the ViT's final LayerNorm."""

    def __init__(self, dim, hidden, out=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                                 nn.Linear(hidden, out or dim))

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    """Action embedder: a 1x1 convolution over the action block, then an MLP."""

    def __init__(self, in_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.embed = nn.Sequential(nn.Linear(in_dim, mlp_scale * emb_dim), nn.SiLU(),
                                   nn.Linear(mlp_scale * emb_dim, emb_dim))

    def forward(self, x):
        """x: [B, T, in_dim] -> [B, T, emb_dim]."""
        return self.embed(self.patch_embed(x.transpose(1, 2)).transpose(1, 2))


class ViTEncoder(nn.Module):
    """ViT over one frame; the [CLS] token of the last layer is the representation."""

    def __init__(self, dim=192, depth=6, heads=3, dim_head=64, mlp_dim=768, patch=8, dropout=0.0):
        super().__init__()
        n_patches = (SIZE // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, frames):
        """frames: [B, 3, SIZE, SIZE] in [0, 1] -> [B, dim]."""
        x = (frames - self.mean) / self.std
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


class ARPredictor(nn.Module):
    """Autoregressive predictor: causal transformer over the embedding history, AdaLN-conditioned
    on the action of each step."""

    def __init__(self, dim=192, depth=3, heads=4, dim_head=48, mlp_dim=512, dropout=0.1, length=HIST):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, length, dim) * 0.02)
        self.blocks = nn.ModuleList(ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, act_emb):
        """z, act_emb: [B, T, dim] -> [B, T, dim]; position t predicts the embedding at t + 1."""
        x = z + self.pos_embedding[:, : z.shape[1]]
        for block in self.blocks:
            x = block(x, act_emb)
        return self.norm(x)


class LeWorldModel(nn.Module):
    """The submission contract: (frames [B, HIST, 3, 64, 64], actions [B, HORIZON, 2*SKIP])
    -> z [B, HORIZON + 1, D], with z[:, 0] the embedding of o[t] and z[:, k] the prediction for
    o[t + SKIP*k]."""

    def __init__(self, dim=192, enc_depth=6, enc_heads=3, enc_dim_head=64, enc_mlp=768, patch=8,
                 pred_depth=3, pred_heads=4, pred_dim_head=48, pred_mlp=512, pred_dropout=0.1,
                 proj_hidden=512, identity_dynamics=False):
        super().__init__()
        self.encoder = ViTEncoder(dim, enc_depth, enc_heads, enc_dim_head, enc_mlp, patch)
        self.projector = MLP(dim, proj_hidden)
        self.action_encoder = Embedder(ACTION_BLOCK, dim)
        self.predictor = ARPredictor(dim, pred_depth, pred_heads, pred_dim_head, pred_mlp, pred_dropout)
        self.pred_proj = MLP(dim, proj_hidden)
        # stands in for the actions before t, which the evaluator does not provide
        self.action_mask = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.identity_dynamics = identity_dynamics
        self.register_buffer("action_mean", torch.zeros(1, 1, ACTION_BLOCK))
        self.register_buffer("action_std", torch.ones(1, 1, ACTION_BLOCK))

    def set_action_stats(self, mean, std):
        """Bake the corpus action statistics in: the evaluator feeds raw actions."""
        self.action_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, 1, ACTION_BLOCK))
        self.action_std.copy_(torch.as_tensor(std, dtype=torch.float32).view(1, 1, ACTION_BLOCK))

    def encode(self, frames):
        """frames: [B, T, 3, SIZE, SIZE] -> z [B, T, D]."""
        b, t = frames.shape[:2]
        h = self.encoder(frames.flatten(0, 1))
        return self.projector(h).view(b, t, -1)

    def embed_actions(self, actions):
        """actions: [B, HORIZON, 2*SKIP] raw -> [B, HORIZON, D]."""
        return self.action_encoder((actions - self.action_mean) / self.action_std)

    def rollout(self, z_hist, act_emb):
        """z_hist: [B, HIST, D] embeddings of o[t-10], o[t-5], o[t]; act_emb: [B, HORIZON, D] the
        embedded action blocks from t on. Returns the HORIZON predictions [B, HORIZON, D].

        Step k attends over the last HIST embeddings and conditions on their actions, which are
        the mask embedding for every position before t.
        """
        b = z_hist.shape[0]
        mask = self.action_mask.expand(b, HIST - 1, -1)
        z_seq = [z_hist[:, k] for k in range(HIST)]
        a_seq = [mask[:, k] for k in range(HIST - 1)] + [act_emb[:, 0]]
        preds = []
        for k in range(act_emb.shape[1]):
            if self.identity_dynamics:  # the floor: keep the actions in the graph, ignore them
                pred = z_seq[-1] + 0 * act_emb[:, k, :1]
            else:
                z = torch.stack(z_seq[-HIST:], 1)
                a = torch.stack(a_seq[-HIST:], 1)
                pred = self.pred_proj(self.predictor(z, a)[:, -1])
            preds.append(pred)
            z_seq.append(pred)
            if k + 1 < act_emb.shape[1]:
                a_seq.append(act_emb[:, k + 1])
        return torch.stack(preds, 1)

    def forward(self, frames, actions):
        z = self.encode(frames)
        preds = self.rollout(z, self.embed_actions(actions))
        return torch.cat([z[:, -1:], preds], 1)


def parameter_report(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    parts = {name: sum(p.numel() for p in m.parameters())
             for name, m in model.named_children() if sum(p.numel() for p in m.parameters())}
    parts = ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in sorted(parts.items(), key=lambda kv: -kv[1]))
    return f"{total / 1e6:.2f}M parameters ({total * 4 / 2**20:.1f} MiB of float32): {parts}"


def check_budget(model: nn.Module, cap_mib: float = 20.0) -> None:
    mib = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    if mib > cap_mib * 0.98:
        raise ValueError(f"float32 weights are {mib:.1f} MiB, too close to the {cap_mib} MiB cap")


def flops_per_sample(model: LeWorldModel) -> float:
    """Rough multiply-accumulate count of one forward pass, to sanity-check the CPU budget."""
    enc = model.encoder
    tokens = (SIZE // enc.patch_embed.kernel_size[0]) ** 2 + 1
    dim = enc.cls_token.shape[-1]
    per_block = 0.0
    for block in enc.blocks:
        inner = block.attn.to_qkv.out_features // 3
        per_block += tokens * dim * 3 * inner + 2 * tokens * tokens * inner + tokens * inner * dim
        per_block += 2 * tokens * dim * block.mlp.net[1].out_features
    patch = enc.patch_embed
    enc_cost = HIST * (per_block + tokens * dim * math.prod(patch.kernel_size) * 3)
    pred_block = 0.0
    for block in model.predictor.blocks:
        inner = block.attn.to_qkv.out_features // 3
        pred_block += HIST * dim * 3 * inner + 2 * HIST * HIST * inner + HIST * inner * dim
        pred_block += 2 * HIST * dim * block.mlp.net[1].out_features + HIST * dim * 6 * dim
    return enc_cost + HORIZON * pred_block
