#!/bin/bash
#SBATCH --job-name=fig_early_pred
#SBATCH --partition=normal
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gen_fig_early_pred_%j.out
#SBATCH --error=logs/gen_fig_early_pred_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.figures.gen_fig_early_pred \
    --oof "$DATA/ost_cv/preds_ost.jsonl.gz" \
    --out figures
