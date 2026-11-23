#!/bin/bash
# Slurm wrapper for the ablation: runs its arguments as a command from the repository root, in the repo's .venv.
# launch.py submits every job through this script and sets the #SBATCH options on the sbatch command line.
#
#   sbatch --chdir=<repo> [options] experiments/ablations/job.sh python experiments/ablations/train.py ...

echo "===== Slurm job info ====="
echo "Job: ${SLURM_JOB_NAME:-} (${SLURM_JOB_ID:-})"
echo "Node(s): ${SLURM_JOB_NODELIST:-} (running on $(hostname))"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null
echo "Command: $*"
echo "=========================="

set -euo pipefail
source .venv/bin/activate   # the existing venv; `uv run` would first re-sync it to uv.lock
unset LD_LIBRARY_PATH       # use .venv's own CUDA libraries, not ones from ~/.bashrc
# Temporary files on the node's local disk, one folder per job, removed at exit. Not on NFS: deleting a folder there
# while a file in it is still open fails ("Directory not empty"), which once crashed a job when Triton compiled its
# CUDA helper at the first training step.
export TMPDIR=$(mktemp -d "/tmp/${USER}_abl_${SLURM_JOB_ID:-local}_XXXX")
trap 'rm -rf "$TMPDIR"' EXIT

"$@"
echo "Finished: $(date)"
