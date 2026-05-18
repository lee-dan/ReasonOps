#!/bin/bash
#SBATCH --job-name=model_id_opxgb
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/model_id_opxgb_%j.out
#SBATCH --error=logs/model_id_opxgb_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

if [ -f .env ]; then source .env; fi

VENV=${REASONOPS_VENV}

mkdir -p logs data/model_id

python -m reasonops.eval.model_id_opxgb \
    --corpus ${REASONOPS_DATA}/final_dataset.jsonl.gz \
    --output data/model_id/model_id_opxgb_predictions.jsonl.gz
