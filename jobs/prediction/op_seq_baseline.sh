#!/bin/bash
#SBATCH --job-name=tfidf
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tfidf_%j.out
#SBATCH --error=logs/tfidf_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.op_seq_baseline \
    --corpus  "$DATA/final_dataset.jsonl.gz" \
    --output  "$DATA/predictions/preds_tfidf.jsonl.gz"
