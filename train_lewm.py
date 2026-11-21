"""Train LeWorldModel (LeWM) on the challenge corpus and export it to ONNX.

    uv run train_lewm.py --corpus corpus --out lewm.onnx --steps 12000

The objective combines latent prediction and SIGReg.

    loss = MSE(predicted embeddings, encoded future embeddings) + lambda * SIGReg(embeddings)

Gradients also flow through the targets. The predictor is rolled out for five steps during
training, with supervision at every step to match the challenge horizon.

``--arch reference`` trains the organizers' example model on exactly the same episode split, so
the two can be compared on held-out episodes (see proxy_eval.py).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import torch

import stable_pretraining as spt

import corpus_loader
import example_train
import lewm
from evaluate import HIST, SKIP
from lewm import LeWorldModel, SIGReg


def action_stats(corpus, episodes):
    """Mean and standard deviation of an action block, over the training episodes."""
    rows = np.concatenate([np.arange(corpus.offsets[i], corpus.offsets[i + 1]) for i in episodes])
    a = np.asarray(corpus.actions[rows]).astype(np.float64)
    mean, std = a.mean(0), a.std(0)
    return np.tile(mean, SKIP), np.tile(np.maximum(std, 1e-6), SKIP)


def lewm_forward(self, batch, stage, lamb):
    """LeWM's loss: multi-step latent prediction MSE + lambda * SIGReg, in float32."""
    frames = example_train.to_model_frames(batch["frames"])  # [B, HIST + HORIZON, 3, 64, 64]
    z = self.backbone.encode(frames).float()  # [B, HIST + HORIZON, D]
    act_emb = self.backbone.embed_actions(batch["actions"]).float()
    pred = self.backbone.rollout(z[:, :HIST], act_emb).float()  # [B, HORIZON, D]
    pred_loss = (pred - z[:, HIST:]).square().mean()
    sigreg = self.sigreg(z.transpose(0, 1))  # [T, B, D]: one statistic per step, averaged
    loss = pred_loss + lamb * sigreg
    self.log_dict({f"{stage}/pred": pred_loss, f"{stage}/sigreg": sigreg,
                   f"{stage}/std": z.flatten(0, 1).std(0).mean()}, prog_bar=True)
    return {"loss": loss}


def build(args):
    if args.arch == "reference":
        return example_train.WorldModel(identity_dynamics=args.floor), example_train.jepa_forward
    model = LeWorldModel(dim=args.dim, enc_depth=args.enc_depth, enc_heads=args.heads,
                         enc_dim_head=args.dim_head, enc_mlp=args.enc_mlp, patch=args.patch,
                         pred_depth=args.pred_depth, pred_heads=args.heads,
                         pred_dim_head=args.dim_head, pred_mlp=args.pred_mlp,
                         proj_hidden=args.proj_hidden, identity_dynamics=args.floor)
    print(lewm.parameter_report(model), file=sys.stderr)
    print(f"~{lewm.flops_per_sample(model) * 2 / 1e9:.2f} GFLOP per sample of inference",
          file=sys.stderr)
    lewm.check_budget(model)
    return model, partial(lewm_forward, lamb=args.lamb)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path, default=Path("lewm.onnx"))
    p.add_argument("--arch", choices=["lewm", "reference"], default="lewm")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lamb", type=float, default=0.1, help="SIGReg weight")
    p.add_argument("--val-frac", type=float, default=0.1, help="episodes held out of training")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=("16-mixed", "bf16-mixed", "32-true"))
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--enc-mlp", type=int, default=512)
    p.add_argument("--pred-depth", type=int, default=6)
    p.add_argument("--pred-mlp", type=int, default=512)
    p.add_argument("--proj-hidden", type=int, default=512)
    p.add_argument("--floor", action="store_true", help="export an untrained, no-dynamics LeWM")
    p.add_argument("--probe", action="store_true", help="only export an untrained model, to check size and speed")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    model, forward = build(args)
    if args.probe or args.floor:
        example_train.export(model, args.out)
        return
    if args.corpus is None:
        p.error("--corpus is required unless --probe or --floor")

    corpus = corpus_loader.load(args.corpus)
    try:
        train_eps, val_eps = corpus_loader.episode_split(corpus, args.val_frac)
    except ValueError as error:
        p.error(str(error))
    print(f"training on {len(train_eps)} episodes, holding out {len(val_eps)}", file=sys.stderr)
    if args.arch == "lewm":
        model.set_action_stats(*action_stats(corpus, train_eps))

    module = spt.Module(
        backbone=model,
        sigreg=SIGReg(),
        vcreg=spt.losses.utils.VCRegLoss(std_coeff=25, cov_coeff=1),  # only used by --arch reference
        forward=forward,
        optim={"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
               "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                             "warmup_steps": max(1, args.steps // 20), "max_steps": args.steps},
               "interval": "step"},
    )
    data = example_train.Windows(corpus, train_eps)
    sampler = torch.utils.data.WeightedRandomSampler(data.weights, num_samples=args.steps * args.batch_size)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    trainer = pl.Trainer(
        max_steps=args.steps,
        precision=args.precision or ("bf16-mixed" if torch.cuda.is_available() else "32-true"),
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loader))()

    example_train.export(module.backbone, args.out)
    print(f"\nNow check it:  uv run test_submission.py {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
