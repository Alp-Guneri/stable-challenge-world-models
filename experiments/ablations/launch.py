"""Submit a suite of combinations to Slurm: one job per combination (its seeds run in series inside the job), one job
for the blend anchors, and one report job that starts when all of them have ended.

    .venv/bin/python experiments/ablations/launch.py --name smoke --steps 10 --anchor-steps 10   # quick format check
    .venv/bin/python experiments/ablations/launch.py --name main                                 # encoder x predictor x loss
    .venv/bin/python experiments/ablations/launch.py --name loss --suite loss --reuse-from main  # CNN-MLP x loss presets
    .venv/bin/python experiments/ablations/launch.py --name main --dry-run                       # print, submit nothing

Suites and loss presets are defined in presets.py; --combos runs an explicit list instead. --reuse-from links
combinations (and anchors) that other runs already finished with the same steps and seeds, instead of retraining
(several names: each combination is taken from the first that has it).
Everything goes to experiments/ablations/results/<name>/ (Slurm logs in its slurm/ folder; job ids in jobs.json).
Other experiments reuse this launcher with their own suites and results folder (experiments/predictor_history).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from presets import ENCODERS, LOSSES, PREDICTORS, SUITES  # noqa: E402
BF16_GPUS = "rtx_a6000|l40|l40s|rtxpro_6000"  # bf16-capable nodes of this cluster (Ampere or newer)
RESULTS = HERE / "results"


def combo(name):
    """argparse type: '<encoder>-<predictor>-<loss>' with known parts."""
    parts = name.split("-")
    if len(parts) != 3 or parts[0] not in ENCODERS or parts[1] not in PREDICTORS or parts[2] not in LOSSES:
        raise argparse.ArgumentTypeError(f"{name!r}: expected <encoder>-<predictor>-<loss> with encoder in "
                                         f"{ENCODERS}, predictor in {PREDICTORS}, loss in {LOSSES}")
    return name


def finished(run_dir, seeds, steps):
    """True if run_dir holds seeds 0..seeds-1, each evaluated (done.json) and trained for `steps` steps."""
    for s in range(seeds):
        d = run_dir / f"seed{s}"
        try:
            if not (d / "done.json").exists() or json.loads((d / "config.json").read_text())["steps"] != steps:
                return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def link(target, source, dry_run):
    """<results>/<name>/<x> -> <results of the other run>/<x> (relative symlink)."""
    print(f"reuse: {target.relative_to(ROOT)} -> {source.relative_to(ROOT)}")
    if not dry_run and not target.exists():
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)


def submit(sbatch_args, command, dry_run):
    cmd = ["sbatch", "--parsable", f"--chdir={ROOT}", *sbatch_args, str(HERE / "job.sh"), *command]
    print(" ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return "DRYRUN"
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def main(argv=None, *, suites=SUITES, results_root=RESULTS, report_script=HERE / "report.py", job_prefix="abl",
         description=__doc__, reuse_default=None):
    """Also the launcher of other experiments: their suites, results folder, report script and Slurm job-name prefix.
    --reuse-from names are looked up in results_root, then in this folder's results."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help=f"results go to {results_root.relative_to(ROOT)}/<name>/")
    p.add_argument("--suite", choices=suites, default=next(iter(suites)),
                   help="; ".join(f"{k}: {' '.join(v)}" for k, v in suites.items()))
    p.add_argument("--combos", nargs="+", type=combo, metavar="COMBO", help="run these instead of the suite")
    p.add_argument("--reuse-from", nargs="+", metavar="NAME", default=reuse_default,
                   help="link combinations and anchors already finished in these runs (same --steps and --seeds; "
                        "the first run that has one wins) instead of training them again"
                        + (f"; default: {' '.join(reuse_default)}" if reuse_default else ""))
    p.add_argument("--no-reuse", action="store_true", help="train everything, ignoring --reuse-from")
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
    args = p.parse_args(argv)

    results = results_root / args.name
    logs = results / "slurm"
    logs.mkdir(parents=True, exist_ok=True)  # sbatch opens the log files before the job starts
    partition = args.partition or ("compute" if args.cpu else "gpus")
    common = [f"--partition={partition}", f"--cpus-per-task={args.cpus}", f"--mem={args.mem}",
              f"--output={logs}/%x_%j.out", f"--error={logs}/%x_%j.err"]
    gpu = [] if args.cpu else ["--gres=gpu:1"] + ([f"--constraint={args.constraint}"] if args.constraint else [])
    python = str(ROOT / ".venv" / "bin" / "python")

    combos = args.combos or suites[args.suite]
    sources = []
    for name in ([] if args.no_reuse else args.reuse_from or []):
        source = next((r / name for r in (results_root, RESULTS) if (r / name).is_dir()), None)
        if source is None:
            sys.exit(f"--reuse-from: no results named {name!r} in {results_root} or {RESULTS}")
        sources.append(source)
    root_args = ["--results-root", str(results_root)]

    jobs, reused = {}, []
    for name in combos:
        source = next((s for s in sources if finished(s / name, args.seeds, args.steps)), None)
        if source is not None:
            link(results / name, source / name, args.dry_run)
            reused.append(name)
            continue
        enc, pred, loss = name.split("-")
        jobs[name] = submit(common + gpu + [f"--job-name={job_prefix}_{name}", f"--time={args.time}"],
                            [python, str(HERE / "train.py"), "--encoder", enc, "--predictor", pred, "--loss", loss,
                             "--seeds", str(args.seeds), "--steps", str(args.steps), "--name", args.name,
                             *root_args, "--workers", str(args.workers), "--log-every", str(args.log_every),
                             "--precision", args.precision]
                            + (["--skip-eval"] if args.skip_eval else []), args.dry_run)
    anchor_source = next((s for s in sources if all((s / "anchors" / a / "proxy.json").exists()
                                                    for a in ("floor", "ref_holdout"))), None)
    if anchor_source is not None and not (args.no_anchors or args.skip_eval):
        link(results / "anchors", anchor_source / "anchors", args.dry_run)
        reused.append("anchors")
    elif not (args.no_anchors or args.skip_eval):
        jobs["anchors"] = submit(common + gpu + [f"--job-name={job_prefix}_anchors", f"--time={args.anchor_time}"],
                                 [python, str(HERE / "anchors.py"), "--name", args.name, *root_args, "--steps",
                                  str(args.anchor_steps), "--workers", str(args.workers),
                                  "--precision", args.precision], args.dry_run)
    # afterany: the report still runs if one job fails, with whatever finished. It needs no GPU, so it goes to the CPU
    # partition: the 'gpus' partition refuses jobs without a GPU (they wait forever with reason QOSMinGRES).
    report_opts = ["--partition=compute", "--cpus-per-task=2", "--mem=16G", f"--output={logs}/%x_%j.out",
                   f"--error={logs}/%x_%j.err", f"--job-name={job_prefix}_report", "--time=1:00:00"]
    jobs["report"] = submit(report_opts + ([f"--dependency=afterany:{':'.join(jobs.values())}"] if jobs else []),
                            [python, str(report_script), "--name", args.name, *root_args], args.dry_run)

    if not args.dry_run:
        (results / "jobs.json").write_text(json.dumps({"jobs": jobs, "reused": reused, "combos": combos,
                                                       "args": vars(args)}, indent=2) + "\n")
    if reused:
        print(f"\nreused from {' '.join(args.reuse_from)}: {' '.join(reused)}")
    print(f"\n{len(jobs)} jobs {'(dry run, nothing submitted)' if args.dry_run else 'submitted'}; "
          f"results in {results}\nwatch them with: squeue -u $USER -n "
          f"{','.join(f'{job_prefix}_{c}' for c in jobs)}")


if __name__ == "__main__":
    main()
