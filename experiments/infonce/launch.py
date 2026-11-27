"""Submit an InfoNCE experiment to Slurm, with the ablation's launcher: one job per combination (its seeds in series
inside the job) and a report job when all have ended. The reference loss, ref_nce2x and the blend anchors are linked
from experiments/ablations/results/{main,loss} instead of retrained.

    .venv/bin/python experiments/infonce/launch.py --name weights --dry-run   # print, submit nothing
    .venv/bin/python experiments/infonce/launch.py --name weights             # the real run

Suites are in infonce_presets.py. Results go to experiments/infonce/results/<name>/. Every option of
experiments/ablations/launch.py works here (--seeds, --steps, --combos, --constraint, --time, ...).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The ablation's folder first: its launch.py, presets.py and report.py, not this folder's same-named scripts.
sys.path.insert(0, str(HERE.parent / "ablations"))
sys.path.insert(1, str(HERE))
import launch as ablation_launch  # noqa: E402  (experiments/ablations/launch.py)
from infonce_presets import REUSE_FROM, SUITES  # noqa: E402

if __name__ == "__main__":
    assert Path(ablation_launch.__file__).parent.name == "ablations"
    ablation_launch.main(suites=SUITES, results_root=HERE / "results", report_script=HERE / "report.py",
                         job_prefix="nce", description=__doc__, reuse_default=REUSE_FROM)
