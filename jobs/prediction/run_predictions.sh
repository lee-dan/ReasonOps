#!/bin/bash
#SBATCH --job-name=run_preds
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/run_predictions_%j.out
#SBATCH --error=logs/run_predictions_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.run_predictions \
    --preds_dir "$DATA/predictions" \
    --ost_dir   "$DATA/ost_cv" \
    --corpus    "$DATA/final_dataset.jsonl.gz" \
    --output    "$DATA/predictions/all_predictions.jsonl.gz"
