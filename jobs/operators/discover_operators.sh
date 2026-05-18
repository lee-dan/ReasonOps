#!/bin/bash
#SBATCH --job-name=discover_ops
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/discover_operators_%j.out
#SBATCH --error=logs/discover_operators_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"
OPS_DIR="$DATA/operators"

python -m reasonops.operators.discover_operators \
    --input      "$DATA/traces_filtered.jsonl.gz" \
    --output_dir "$OPS_DIR" \
    --config     configs/operators.toml
