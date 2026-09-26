#!/bin/bash
#SBATCH --job-name=op_xgb_early
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/op_xgb_early_%j.out
#SBATCH --error=logs/op_xgb_early_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

if [ -f .env ]; then source .env; fi

VENV=${REASONOPS_VENV}
DATA=${REASONOPS_DATA}

mkdir -p logs data/predictions

"$VENV/bin/python3" -m reasonops.prediction.op_xgb_early \
    --corpus "$DATA/final_dataset.jsonl.gz" \
    --output data/predictions/preds_opxgb_early.jsonl.gz \
    --summary data/predictions/opxgb_early_summary.json
