"""Train a larger JEPA world model on the challenge corpus and export it to ONNX.

The objective is the organizers' reference loss (latent MSE, InfoNCE, VCReg). The encoder is the
part that changes: the reference squeezes each stack of frames down to a 4x4 map, which throws
away where things are. This one keeps an 8x8 map, tells the first layer the pixel coordinates,
and feeds it the frame-to-frame differences so velocity does not have to be learned as a
subtraction. The predictor is the same residual MLP, widened.

    uv run train_strong.py --corpus corpus --out strong.onnx --steps 30000
    uv run train_strong.py --probe --out /tmp/strong_probe.onnx
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining.backbone import MLP
from stable_pretraining.losses.utils import VCRegLoss

import corpus_loader
import example_train
from evaluate import ACTION_DIM, HIST, HORIZON, SIZE, SKIP
from train_lewm import action_stats


def groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        g = groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        return self.act(x + self.norm2(self.conv2(h)))


class Encoder(nn.Module):
    """Stacked frames, their differences, and a coordinate grid -> one embedding.

    Three stride-2 convolutions land on an 8x8 map (the reference goes on to 4x4). A 2x2
    average of the 16x16 map is concatenated with that 8x8 map so a linear probe can still
    read off which quadrant something is in.
    """

    def __init__(self, dim=512, channels=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = channels
        in_ch = HIST * 3 + (HIST - 1) * 3 + 2
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, c1, 4, stride=2, padding=1),
                                   nn.GroupNorm(groups(c1), c1), nn.GELU(), ResBlock(c1))
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, 4, stride=2, padding=1),
                                   nn.GroupNorm(groups(c2), c2), nn.GELU(), ResBlock(c2))
        self.down3 = nn.Sequential(nn.Conv2d(c2, c3, 4, stride=2, padding=1),
                                   nn.GroupNorm(groups(c3), c3), nn.GELU(), ResBlock(c3), ResBlock(c3))
        flat = c3 * (SIZE // 8) ** 2 + c2 * 4
        self.proj = nn.Linear(flat, dim)
        ys = torch.linspace(-1, 1, SIZE)
        xs = torch.linspace(-1, 1, SIZE)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("coord", torch.stack([gx, gy])[None])

    def forward(self, frames):
        """frames [B, HIST, 3, 64, 64] in [0, 1] -> [B, dim]."""
        x = frames - 0.5
        deltas = [x[:, i + 1] - x[:, i] for i in range(HIST - 1)]
        h = torch.cat([x.flatten(1, 2), *deltas, self.coord.expand(x.shape[0], -1, -1, -1)], 1)
        h = self.down1(h)
        h16 = self.down2(h)
        quadrants = F.avg_pool2d(h16, h16.shape[-1] // 2).flatten(1)
        h8 = self.down3(h16).flatten(1)
        return self.proj(torch.cat([h8, quadrants], 1))


class Predictor(nn.Module):
    """Residual one-step dynamics. Actions are normalized with the corpus statistics."""

    def __init__(self, dim=512, hidden=768):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim + ACTION_DIM * SKIP, [hidden, hidden, dim], activation_layer=nn.GELU)
        self.register_buffer("action_mean", torch.zeros(1, ACTION_DIM * SKIP))
        self.register_buffer("action_std", torch.ones(1, ACTION_DIM * SKIP))

    def set_action_stats(self, mean, std):
        self.action_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, z, a):
        a = (a - self.action_mean) / self.action_std
        return z + self.mlp(torch.cat([self.norm(z), a], -1))


class WorldModel(nn.Module):
    def __init__(self, dim=512, channels=(32, 64, 64), hidden=768):
        super().__init__()
        self.encoder = Encoder(dim, channels)
        self.predictor = Predictor(dim, hidden)

    def rollout(self, z, actions):
        out = [z]
        for k in range(actions.shape[1]):
            z = self.predictor(z, actions[:, k])
            out.append(z)
        return torch.stack(out, 1)

    def forward(self, frames, actions):
        return self.rollout(self.encoder(frames), actions)


def describe(model):
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.2f}M parameters, {n * 4 / 2**20:.1f} MiB of float32 weights"


def horizon_weights(device):
    """Later rollout steps are the ones the reference gets wrong, so they count more."""
    w = torch.tensor([1.0, 1.15, 1.35, 1.6, 1.9], device=device)
    return w / w.mean()


def strong_forward(self, batch, stage):
    frames = example_train.to_model_frames(batch["frames"])
    b = frames.shape[0]
    windows = torch.stack([frames[:, k : k + HIST] for k in range(HORIZON + 1)], 1)
    target = self.backbone.encoder(windows.flatten(0, 1)).view(b, HORIZON + 1, -1).float()
    pred = self.backbone.rollout(target[:, 0], batch["actions"])[:, 1:].float()
    w = horizon_weights(pred.device)
    per_step = (pred - target[:, 1:]).square().mean(-1)
    pred_loss = (per_step * w).mean()
    reg = self.vcreg(target.flatten(0, 1))
    reg_t = self.vcreg((target - target.mean(1, keepdim=True)).flatten(0, 1))
    nce = sum(w[k] * example_train.info_nce(pred[:, k], target[:, k + 1]) for k in range(HORIZON)) / w.sum()
    loss = 25 * pred_loss + reg + reg_t + nce
    if getattr(self, "motion_head", None) is not None:
        motion = example_train.motion_loss(self.motion_head, frames, target[:, 1:], pred)
        loss = loss + self.motion_loss_weight * motion
        self.log(f"{stage}/motion", motion, prog_bar=True)
    self.log_dict({f"{stage}/pred": pred_loss, f"{stage}/vcreg": reg, f"{stage}/vcreg_t": reg_t,
                   f"{stage}/nce": nce, f"{stage}/std": target.flatten(0, 1).std(0).mean()}, prog_bar=True)
    return {"loss": loss}


class Snapshot(pl.Callback):
    """Export the contract ONNX every ``every`` steps without ending the run."""

    def __init__(self, every, directory):
        self.every = every
        self.directory = Path(directory)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.every and step % self.every == 0:
            path = self.directory / f"strong_s{step}.onnx"
            example_train.export(copy.deepcopy(module.backbone), path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path, default=Path("strong.onnx"))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--channels", type=int, nargs=3, default=(32, 64, 64))
    p.add_argument("--snapshot-every", type=int, default=10000)
    p.add_argument("--snapshot-dir", type=Path, default=Path("/tmp/strong_snapshots"))
    p.add_argument("--motion-loss-weight", type=float, default=0.0)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    if args.motion_loss_weight < 0:
        p.error("--motion-loss-weight must be nonnegative")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    model = WorldModel(args.dim, tuple(args.channels), args.hidden)
    print(describe(model), file=sys.stderr)
    if args.probe:
        example_train.export(model, args.out)
        return
    if args.corpus is None:
        p.error("--corpus is required unless --probe")

    corpus = corpus_loader.load(args.corpus)
    try:
        train_eps, val_eps = corpus_loader.episode_split(corpus, args.val_frac)
    except ValueError as error:
        p.error(str(error))
    print(f"training on {len(train_eps)} episodes, holding out {len(val_eps)}", file=sys.stderr)
    model.predictor.set_action_stats(*action_stats(corpus, train_eps))

    module = spt.Module(
        backbone=model,
        vcreg=VCRegLoss(std_coeff=25, cov_coeff=1),
        forward=strong_forward,
        optim={"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
               "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                             "warmup_steps": max(1, args.steps // 20), "max_steps": args.steps},
               "interval": "step"},
    )
    if args.motion_loss_weight > 0:
        # Preserve the reference run's random window stream when initializing the auxiliary head.
        with torch.random.fork_rng(devices=[]):
            module.motion_head = example_train.make_motion_head(args.dim)
        module.motion_loss_weight = args.motion_loss_weight
    data = example_train.Windows(corpus, train_eps)
    sampler = torch.utils.data.WeightedRandomSampler(data.weights, num_samples=args.steps * args.batch_size)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        max_steps=args.steps,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        callbacks=[Snapshot(args.snapshot_every, args.snapshot_dir)],
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loader))()
    example_train.export(module.backbone, args.out)


if __name__ == "__main__":
    main()
