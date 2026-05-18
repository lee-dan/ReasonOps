#!/bin/bash
#SBATCH --job-name=temporal
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/temporal_dynamics_%j.out
#SBATCH --error=logs/temporal_dynamics_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.analysis.temporal_dynamics \
    --corpus     "$DATA/final_dataset.jsonl.gz" \
    --output_dir "$DATA/figures/" \
    --k_folds    5
