#!/bin/bash
#SBATCH --job-name=naming_stab
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/naming_stability_%j.out
#SBATCH --error=logs/naming_stability_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"
OPS_DIR="$DATA/operators"

python -m reasonops.eval.naming_stability \
    --input         "$OPS_DIR/spans_clustered.jsonl" \
    --viz            "$OPS_DIR/umap_viz.jsonl" \
    --centroids     "$OPS_DIR/umap_cluster.npy" \
    --vocab         "$OPS_DIR/operator_vocab.txt" \
    --names         "$OPS_DIR/cluster_names.json" \
    --output_dir    "$OPS_DIR/validation/"
