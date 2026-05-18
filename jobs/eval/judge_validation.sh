#!/bin/bash
#SBATCH --job-name=judge_val
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/judge_validation_%j.out
#SBATCH --error=logs/judge_validation_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"
OPS_DIR="$DATA/operators"

python -m reasonops.eval.judge_validation \
    --names         "$OPS_DIR/cluster_names.json" \
    --spans         "$OPS_DIR/spans_clustered.jsonl" \
    --output        "$OPS_DIR/judge_validation.json" \
    --judge         "openai/gpt-4o-mini" \
    --n_per_cluster 50 \
    --workers       8
