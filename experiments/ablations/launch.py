"""Submit the whole ablation to Slurm: one job per combination (its seeds run in series inside the job), one job for
the blend anchors, and one report job that starts when all of them have ended.

    .venv/bin/python experiments/ablations/launch.py --name smoke --steps 10 --anchor-steps 10   # quick format check
    .venv/bin/python experiments/ablations/launch.py --name main                                 # the real run
    .venv/bin/python experiments/ablations/launch.py --name main --dry-run                       # print, submit nothing

Everything goes to experiments/ablations/results/<name>/ (Slurm logs in its slurm/ folder; job ids in jobs.json).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMBOS = [f"{e}-{p}-{l}" for e in ("cnn", "vit") for p in ("mlp", "ar") for l in ("ref", "lewm")] + ["vit-mlp1-ref"]
BF16_GPUS = "rtx_a6000|l40|l40s|rtxpro_6000"  # bf16-capable nodes of this cluster (Ampere or newer)


def submit(sbatch_args, command, dry_run):
    cmd = ["sbatch", "--parsable", f"--chdir={ROOT}", *sbatch_args, str(HERE / "job.sh"), *command]
    print(" ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return "DRYRUN"
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="results go to experiments/ablations/results/<name>/")
    p.add_argument("--combos", nargs="+", default=COMBOS, choices=COMBOS, metavar="COMBO",
                   help=f"default: all {len(COMBOS)}: {' '.join(COMBOS)}")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=32768)
    p.add_argument("--anchor-steps", type=int, default=6000, help="training steps of the blend's reference anchor")
    p.add_argument("--no-anchors", action="store_true", help="skip the anchor job (no blend score)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--skip-eval", action="store_true",
                   help="train and plot only: no test_submission / proxy_eval / action_diagnostic (and no anchors)")
    p.add_argument("--cpu", action="store_true",
                   help="run on CPU nodes (partition 'compute', no GPU; training falls back to float32)")
    p.add_argument("--partition", default=None, help="default: 'gpus', or 'compute' with --cpu")
    p.add_argument("--constraint", default="",
                   help=f"GPU node features; default any GPU node (bf16-capable ones: '{BF16_GPUS}')")
    p.add_argument("--precision", choices=("32-true", "bf16-mixed", "16-mixed"), default="32-true",
                   help="for every training run, anchors included; bf16-mixed needs --constraint with bf16 nodes")
    p.add_argument("--time", default="2-00:00:00", help="per combination job (all its seeds)")
    p.add_argument("--anchor-time", default="6:00:00")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="48G")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    results = HERE / "results" / args.name
    logs = results / "slurm"
    logs.mkdir(parents=True, exist_ok=True)  # sbatch opens the log files before the job starts
    partition = args.partition or ("compute" if args.cpu else "gpus")
    common = [f"--partition={partition}", f"--cpus-per-task={args.cpus}", f"--mem={args.mem}",
              f"--output={logs}/%x_%j.out", f"--error={logs}/%x_%j.err"]
    gpu = [] if args.cpu else ["--gres=gpu:1"] + ([f"--constraint={args.constraint}"] if args.constraint else [])
    python = str(ROOT / ".venv" / "bin" / "python")

    jobs = {}
    for combo in args.combos:
        enc, pred, loss = combo.split("-")
        jobs[combo] = submit(common + gpu + [f"--job-name=abl_{combo}", f"--time={args.time}"],
                             [python, str(HERE / "train.py"), "--encoder", enc, "--predictor", pred, "--loss", loss,
                              "--seeds", str(args.seeds), "--steps", str(args.steps), "--name", args.name,
                              "--workers", str(args.workers), "--log-every", str(args.log_every),
                              "--precision", args.precision]
                             + (["--skip-eval"] if args.skip_eval else []), args.dry_run)
    if not (args.no_anchors or args.skip_eval):
        jobs["anchors"] = submit(common + gpu + ["--job-name=abl_anchors", f"--time={args.anchor_time}"],
                                 [python, str(HERE / "anchors.py"), "--name", args.name, "--steps",
                                  str(args.anchor_steps), "--workers", str(args.workers),
                                  "--precision", args.precision], args.dry_run)
    # afterany: the report still runs if one job fails, with whatever finished. It needs no GPU, so it goes to the CPU
    # partition: the 'gpus' partition refuses jobs without a GPU (they wait forever with reason QOSMinGRES).
    report_opts = ["--partition=compute", "--cpus-per-task=2", "--mem=16G", f"--output={logs}/%x_%j.out",
                   f"--error={logs}/%x_%j.err", "--job-name=abl_report", "--time=1:00:00"]
    jobs["report"] = submit(report_opts + ([f"--dependency=afterany:{':'.join(jobs.values())}"] if jobs else []),
                            [python, str(HERE / "report.py"), "--name", args.name], args.dry_run)

    if not args.dry_run:
        (results / "jobs.json").write_text(json.dumps({"jobs": jobs, "args": vars(args)}, indent=2) + "\n")
    print(f"\n{len(jobs)} jobs {'(dry run, nothing submitted)' if args.dry_run else 'submitted'}; "
          f"results in {results}\nwatch them with: squeue -u $USER -n {','.join('abl_' + c for c in jobs)}")


if __name__ == "__main__":
    main()
