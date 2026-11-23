"""Build and evaluate the two models that anchor the local blend score (as in the repo README):

    floor        example_train.py --floor                       blend 0
    ref_holdout  the reference trained 6k steps on the holdout  blend 1
                 (train_lewm.py --arch reference --lr 1e-3 --weight-decay 1e-4, the README's command)

    .venv/bin/python experiments/ablations/anchors.py --name main [--steps 6000]

Writes results/<name>/anchors/{floor,ref_holdout}/{model.onnx,proxy.json,proxy_eval.txt,train.txt}.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(cmd, log):
    with open(log, "w") as f:
        code = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f"failed ({code}): {' '.join(map(str, cmd))}; see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="default")
    p.add_argument("--steps", type=int, default=6000, help="training steps of ref_holdout (README: 6000)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--precision", choices=("32-true", "bf16-mixed", "16-mixed"), default="32-true")
    p.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    args = p.parse_args()
    out = HERE / "results" / args.name / "anchors"
    floor, ref = out / "floor", out / "ref_holdout"
    for d in (floor, ref):
        d.mkdir(parents=True, exist_ok=True)

    run([ROOT / "example_train.py", "--floor", "--out", floor / "model.onnx"], floor / "train.txt")
    run([ROOT / "train_lewm.py", "--corpus", args.corpus, "--arch", "reference", "--steps", args.steps,
         "--lr", "1e-3", "--weight-decay", "1e-4", "--val-frac", "0.1", "--workers", args.workers,
         "--precision", args.precision, "--out", ref / "model.onnx"], ref / "train.txt")
    for d in (floor, ref):
        run([HERE / "run_ort.py", ROOT / "proxy_eval.py", d / "model.onnx", "--corpus", args.corpus,
             "--val-frac", "0.1", "--out", d / "proxy.json"], d / "proxy_eval.txt")
    print(f"anchors written to {out}")


if __name__ == "__main__":
    main()
