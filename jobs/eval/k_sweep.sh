#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/k_sweep_%j.out
#SBATCH --error=logs/k_sweep_%j.err

set -euo pipefail
source ~/.bashrc

VENV=${REASONOPS_VENV}
DATA=${REASONOPS_DATA}
cd "$(dirname "$0")/../.."

source "$VENV/bin/activate"

# Load API key from .env if present
if [ -f .env ]; then source .env; fi

mkdir -p logs data/k_sweep

python -m reasonops.eval.k_sweep \
    --spans "$DATA/operators/discovered_k7/spans_clustered.jsonl" \
    --out   data/k_sweep \
    --k_values 6 8 9 10 11 \
    --n_per_cluster 50
