#!/bin/bash
#SBATCH --job-name=seq_ost_w
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/seq_pred_ost_within_%j.out
#SBATCH --error=logs/seq_pred_ost_within_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.seq_pred \
    --mode       within \
    --corpus     "$DATA/final_dataset.jsonl.gz" \
    --output_dir "$DATA/ost_cv" \
    --epochs     60 \
    --patience   12
