"""Submit the predictor-history experiment to Slurm, with the ablation's launcher: one job per combination (its seeds
in series inside the job) and a report job when all have ended. The baselines, cnn-ar-ref and the blend anchors are
linked from experiments/ablations/results/{main,encoder} instead of retrained.

    .venv/bin/python experiments/predictor_history/launch.py --name all --dry-run         # print, submit nothing
    .venv/bin/python experiments/predictor_history/launch.py --name smoke --steps 10 --anchor-steps 10
    .venv/bin/python experiments/predictor_history/launch.py --name all                   # the real run
    .venv/bin/python experiments/predictor_history/launch.py --name opt3 --suite option3  # one option only

Suites are in history_presets.py. Results go to experiments/predictor_history/results/<name>/. Every option of
experiments/ablations/launch.py works here (--seeds, --steps, --combos, --constraint, --time, ...).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The ablation's folder first: its launch.py, presets.py and report.py, not this folder's same-named scripts.
sys.path.insert(0, str(HERE.parent / "ablations"))
sys.path.insert(1, str(HERE))
import launch as ablation_launch  # noqa: E402  (experiments/ablations/launch.py)
from history_presets import REUSE_FROM, SUITES  # noqa: E402

if __name__ == "__main__":
    assert Path(ablation_launch.__file__).parent.name == "ablations"
    ablation_launch.main(suites=SUITES, results_root=HERE / "results", report_script=HERE / "report.py",
                         job_prefix="ph", description=__doc__, reuse_default=REUSE_FROM)
