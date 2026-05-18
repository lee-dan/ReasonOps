#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

mkdir -p logs figures figures/model_id

python -m reasonops.figures.gen_fig_model_id_confusion \
    --preds "$DATA/model_id/model_id_opxgb_predictions.jsonl.gz" \
    --out figures

python -m reasonops.figures.gen_fig_model_id_barplot \
    --preds "$DATA/model_id/model_id_opxgb_predictions.jsonl.gz" \
    --out figures/model_id

python -m reasonops.figures.gen_fig_dataset_heatmap_correctness \
    --corpus "$DATA/final_dataset.jsonl.gz" \
    --names  "$DATA/cluster_names.json" \
    --out figures

python -m reasonops.figures.gen_fig_stability \
    --out figures
