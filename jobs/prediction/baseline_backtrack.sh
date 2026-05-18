#!/bin/bash
#SBATCH --job-name=bl_backtrack
#SBATCH --partition=normal
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/baseline_backtrack_%j.out
#SBATCH --error=logs/baseline_backtrack_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.baseline_backtrack \
    --corpus "$DATA/final_dataset.jsonl.gz" \
    --names  "$DATA/operators/cluster_names.json" \
    --output "$DATA/predictions/preds_backtrack.jsonl.gz"
