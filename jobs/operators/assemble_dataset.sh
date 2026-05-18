#!/bin/bash
#SBATCH --job-name=assemble
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/assemble_dataset_%j.out
#SBATCH --error=logs/assemble_dataset_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"
OPS_DIR="$DATA/operators"
TRACES_DIR="$DATA/traces"

python -m reasonops.operators.assemble_dataset \
    --traces_dir    "$TRACES_DIR" \
    --spans         "$OPS_DIR/spans_clustered.jsonl" \
    --names         "$OPS_DIR/cluster_names.json" \
    --output        "$DATA/final_dataset.jsonl.gz"
