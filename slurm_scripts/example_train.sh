#!/bin/bash

#SBATCH --job-name=wm_example_train
#SBATCH --output=/cs/data/people/svangaru/stable-challenge-world-models/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/stable-challenge-world-models/slurm_logs/%x_%j.err

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G

# Usage (from anywhere):
#   sbatch slurm_scripts/train_example.sh                      # reference run, writes runs/example_<jobid>.onnx
#   sbatch slurm_scripts/train_example.sh --seed 1 --steps 12000   # extra args go to example_train.py
# The slurm_logs/ directory must exist before submitting (sbatch opens the log files first).

# Print Slurm job info
echo "===== Slurm job info ====="
echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node(s): ${SLURM_JOB_NODELIST} (running on $(hostname))"
echo "Started: $(date)"
echo "SBATCH directives:"
grep '^#SBATCH' "$0" | sed 's/^#SBATCH /    /'
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null
echo "Extra args: $*"
echo "=========================="

set -euo pipefail

REPO=/cs/data/people/svangaru/stable-challenge-world-models
cd "$REPO"
source .venv/bin/activate  # created by `uv sync`
unset LD_LIBRARY_PATH      # use .venv's own CUDA libs, not text-planning's from ~/.bashrc
export TMPDIR=/cs/data/people/svangaru/tmp  # Node-local /ltmp can be full
mkdir -p "$TMPDIR" runs

OUT="runs/example_${SLURM_JOB_ID}.onnx"

# Train the reference JEPA world model (~7 min on a laptop GPU) and export it to ONNX.
# DataLoader workers = cpus-per-task minus 2 (main process + headroom).
python example_train.py \
    --corpus corpus \
    --out "$OUT" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-10} - 2 )) \
    "$@"

# Validate the export with the evaluator's own loading/inference code.
python test_submission.py "$OUT"

echo "Finished: $(date)"
