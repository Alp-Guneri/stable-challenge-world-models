"""Train one encoder x predictor x loss combination for several seeds, one after the other, then evaluate and plot.

    .venv/bin/python experiments/ablations/train.py --encoder vit --predictor ar --loss lewm --seeds 3 --name main

Each seed writes <results-root>/<name>/<encoder>-<predictor>-<loss>/seed<k>/ (default results-root:
experiments/ablations/results; see README.md). A seed
whose done.json exists is skipped, so a resubmitted job resumes where it stopped.

Optimizers: one AdamW per network (stable-pretraining assigns parameters by module name), each with the settings of
the model the network comes from (models.OPTIM): learning rate, weight decay, schedule and gradient clipping. The
loss has no optimizer settings of its own.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger

import stable_pretraining as spt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models  # noqa: E402  (puts the repo root on sys.path)
from models import ROOT  # noqa: E402

import corpus_loader  # noqa: E402
import example_train  # noqa: E402
import lewm  # noqa: E402
from train_lewm import action_stats  # noqa: E402

RESULTS = HERE / "results"


def combo_name(encoder, predictor, loss):
    return f"{encoder}-{predictor}-{loss}"


def scheduler(kind, steps):
    if kind == "cosine":  # example_train.py
        return {"type": "CosineAnnealingLR", "T_max": steps}
    return {"type": "LinearWarmupCosineAnnealingLR", "warmup_steps": max(1, steps // 20), "max_steps": steps}


def optim_config(encoder, predictor, steps):
    cfg = {}
    for group, kind in (("encoder", encoder), ("predictor", predictor)):
        o = models.OPTIM[kind]
        cfg[group] = {"modules": rf"backbone\.{group}$",
                      "optimizer": {"type": "AdamW", "lr": o["lr"], "weight_decay": o["weight_decay"]},
                      "scheduler": scheduler(o["schedule"], steps), "interval": "step", "frequency": 1}
    return cfg


def ablation_forward(self, batch, stage, loss_name):
    loss, terms = models.loss_terms(self.backbone, batch, self.vcreg, self.sigreg, loss_name)
    self.log_dict({f"train/{k}": v for k, v in terms.items()}, on_step=True, on_epoch=False)
    return {"loss": loss}


class Diagnostics(pl.Callback):
    """Per-step gradient norm of each network (before clipping), seconds per step, and a progress line."""

    def __init__(self, steps, every=500):
        self.steps, self.every, self.t0, self.start = steps, every, None, time.time()

    def on_after_backward(self, trainer, module):
        for group in ("encoder", "predictor"):
            grads = [p.grad.float().norm() for p in getattr(module.backbone, group).parameters() if p.grad is not None]
            if grads:
                module.log(f"train/grad_norm_{group}", torch.stack(grads).norm(), on_step=True, on_epoch=False)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        module.log("train/step_seconds", time.perf_counter() - self.t0, on_step=True, on_epoch=False)
        step = trainer.global_step
        if step % self.every == 0 or step == self.steps:
            loss = trainer.callback_metrics.get("train/loss")
            print(f"  step {step}/{self.steps}  loss {float(loss) if loss is not None else float('nan'):.4f}  "
                  f"{(time.time() - self.start) / 60:.1f} min", file=sys.stderr, flush=True)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def run_eval(script, args, log_path):
    """Run a repo evaluation script with the venv's python; onnxruntime is forced to CPU when no GPU is visible
    (onnxruntime-gpu crashes on GPU-less nodes). Returns the exit code; output goes to log_path."""
    cmd = [sys.executable, str(HERE / "run_ort.py"), str(ROOT / script), *map(str, args)]
    with open(log_path, "w") as log:
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def train_seed(args, seed, run_dir, corpus, train_eps):
    pl.seed_everything(seed, workers=True)
    model = models.WorldModel(args.encoder, args.predictor)
    if hasattr(model.predictor, "set_action_stats"):  # the AR predictors normalize actions with corpus statistics
        model.predictor.set_action_stats(*action_stats(corpus, train_eps))
    counts = {g: sum(p.numel() for p in getattr(model, g).parameters()) for g in ("encoder", "predictor")}
    mib = sum(t.numel() for t in model.state_dict().values()) * 4 / 2**20
    lewm.check_budget(model)
    print(f"[{run_dir.parent.name} seed {seed}] encoder {counts['encoder'] / 1e6:.2f}M, predictor "
          f"{counts['predictor'] / 1e6:.2f}M parameters, ~{mib:.1f} MiB", file=sys.stderr, flush=True)

    module = spt.Module(backbone=model,
                        vcreg=spt.losses.utils.VCRegLoss(std_coeff=25, cov_coeff=1),
                        sigreg=lewm.SIGReg(),
                        forward=partial(ablation_forward, loss_name=args.loss),
                        optim=optim_config(args.encoder, args.predictor, args.steps))
    # Per-network gradient clipping (stable-pretraining reads these before falling back to the Trainer's value).
    # Note LeWM clips the norm of all its gradients together; here each network's gradients are clipped separately.
    for group, kind in (("encoder", args.encoder), ("predictor", args.predictor)):
        module._optimizer_gradient_clip_val[group] = models.OPTIM[kind]["clip"]
        module._optimizer_gradient_clip_algorithm[group] = "norm"

    data = example_train.Windows(corpus, train_eps)
    generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(data.weights, num_samples=args.steps * args.batch_size,
                                                     generator=generator)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0)
    precision = args.precision
    # stable-pretraining's Manager writes its run folder (logger output, run metadata, a last.ckpt) under its cache
    # dir; point that into this run's folder so nothing lands in ~/.cache.
    spt.set(cache_dir=str(run_dir / "spt"))
    logger = CSVLogger(save_dir=str(run_dir), name="", version="logs", flush_logs_every_n_steps=500)
    trainer = pl.Trainer(max_steps=args.steps, precision=precision, logger=logger, enable_checkpointing=False,
                         log_every_n_steps=args.log_every, enable_progress_bar=False,
                         callbacks=[Diagnostics(args.steps)])
    start = time.time()
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loader))()
    seconds = time.time() - start

    shutil.copy(Path(trainer.logger.log_dir) / "metrics.csv", run_dir / "metrics.csv")
    for ckpt_dir in (run_dir / "spt").rglob("checkpoints"):  # optimizer-state checkpoints; weights.pt is kept instead
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    torch.save(model.state_dict(), run_dir / "weights.pt")
    example_train.export(model, run_dir / "model.onnx")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    config = {
        "combo": run_dir.parent.name, "encoder": args.encoder, "predictor": args.predictor, "loss": args.loss,
        "seed": seed, "steps": args.steps, "batch_size": args.batch_size, "val_frac": args.val_frac,
        "train_episodes": len(train_eps), "precision": precision, "loss_weights": models.LOSS_PRESETS[args.loss],
        "encoder_config": models.encoder_config(args.encoder),
        "optim": {"encoder": models.OPTIM[args.encoder], "predictor": models.OPTIM[args.predictor]},
        "ref_long": models.REF_LONG, "lewm_long": models.LEWM_LONG,
        "parameters": counts, "float32_mib": round(mib, 2), "onnx_mib": round(os.path.getsize(run_dir / "model.onnx") / 2**20, 2),
        "train_seconds": round(seconds, 1), "host": socket.gethostname(), "gpu": gpu,
        "torch": torch.__version__, "python": platform.python_version(), "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def evaluate_seed(args, run_dir):
    onnx = run_dir / "model.onnx"
    codes = {
        "test_submission": run_eval("test_submission.py", [onnx], run_dir / "test_submission.txt"),
        "proxy_eval": run_eval("proxy_eval.py", [onnx, "--corpus", args.corpus, "--val-frac", args.val_frac,
                                                 "--out", run_dir / "proxy.json"], run_dir / "proxy_eval.txt"),
        "action_diagnostic": run_eval("action_diagnostic.py", [onnx, "--corpus", args.corpus,
                                                               "--out", run_dir / "action.json"],
                                      run_dir / "action_diagnostic.txt"),
    }
    print(f"  evaluation exit codes: {codes}", file=sys.stderr, flush=True)
    return codes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", choices=models.ENCODERS, required=True)
    p.add_argument("--predictor", choices=models.PREDICTORS, required=True)
    p.add_argument("--loss", choices=models.LOSSES, required=True)
    p.add_argument("--seeds", type=int, default=3, help="seeds 0 .. N-1 (or from --first-seed), run in series")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1, help="held-out episodes, as proxy_eval.py")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--precision", choices=("32-true", "bf16-mixed", "16-mixed"), default="32-true",
                   help="float32 by default, the same on every GPU (ref_long and lewm_long used bf16-mixed)")
    p.add_argument("--log-every", type=int, default=1, help="log the training metrics every N steps")
    p.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    p.add_argument("--name", default="default", help="results go to <results-root>/<name>/")
    p.add_argument("--results-root", type=Path, default=RESULTS,
                   help="default experiments/ablations/results (experiments/predictor_history uses its own)")
    p.add_argument("--skip-eval", action="store_true", help="skip test_submission / proxy_eval / action_diagnostic")
    args = p.parse_args()

    combo_dir = args.results_root / args.name / combo_name(args.encoder, args.predictor, args.loss)
    corpus = corpus_loader.load(args.corpus)
    train_eps, _ = corpus_loader.episode_split(corpus, args.val_frac)
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        run_dir = combo_dir / f"seed{seed}"
        if (run_dir / "done.json").exists():
            print(f"[{combo_dir.name} seed {seed}] already done, skipping", file=sys.stderr)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        config = train_seed(args, seed, run_dir, corpus, train_eps)
        codes = {} if args.skip_eval else evaluate_seed(args, run_dir)
        (run_dir / "done.json").write_text(json.dumps({"eval_exit_codes": codes,
                                                       "train_seconds": config["train_seconds"]}) + "\n")

    import report  # noqa: E402  (after training: matplotlib is not needed before)
    report.combo_report(combo_dir)


if __name__ == "__main__":
    main()
