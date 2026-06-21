#!/usr/bin/env bash
#SBATCH --job-name=hypclr_pretrain
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=shard:40
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/slurm_%x_%j.out
#SBATCH --error=/tmp/slurm_%x_%j.err

set -euo pipefail

VENV="/N/schwer/venvs/hypclr"
PROJECT="/N/schwer/HypSkeletonCLR_SupCon"
TASK="${TASK:-pretrain_skeletonclr}"
CONFIG="${CONFIG:-config/SkeletonCLR/skeletonclr_xview.yaml}"

source "$VENV/bin/activate"
cd "$PROJECT"
python -u main.py "$TASK" --config "$CONFIG"
