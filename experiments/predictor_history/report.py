"""Plots and tables for the predictor-history experiment: the ablation's report with this experiment's panels (one per
option and encoder family, each against its baseline) and colors.

    .venv/bin/python experiments/predictor_history/report.py --name all

Writes experiments/predictor_history/results/<name>/report/ (same files as experiments/ablations/report.py, the
grouped figures being training_by_arch_ref.png and top1_by_horizon_by_arch_ref.png).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The ablation's folder first: its report.py and presets.py, not this script.
sys.path.insert(0, str(HERE.parent / "ablations"))
sys.path.insert(1, str(HERE))
import report as ablation_report  # noqa: E402  (experiments/ablations/report.py)
from history_presets import ARCH_GROUPS, ARCH_STYLE, SUITES  # noqa: E402


def main():
    assert Path(ablation_report.__file__).parent.name == "ablations"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="<results-root>/<name>/")
    p.add_argument("--results-root", type=Path, default=HERE / "results")
    p.add_argument("--smooth", type=int, default=None, help="rolling-mean window in steps (default: run length/200)")
    args = p.parse_args()
    ablation_report.report(args.results_root / args.name, args.smooth, arch_style=ARCH_STYLE,
                           arch_groups=ARCH_GROUPS, combo_order=SUITES["all"])


if __name__ == "__main__":
    main()
