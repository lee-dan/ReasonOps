#!/bin/bash
#SBATCH --job-name=op_dist
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/operator_distributions_%j.out
#SBATCH --error=logs/operator_distributions_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.analysis.operator_distributions \
    --corpus     "$DATA/final_dataset.jsonl.gz" \
    --names      "$DATA/operators/cluster_names.json" \
    --output_dir "$DATA/figures/"
