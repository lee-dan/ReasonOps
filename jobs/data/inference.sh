#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

TRACES_DIR="${REASONOPS_DATA}/traces"

# Run all models × all datasets (resumes automatically)
python -m reasonops.data.run_inference \
    --all \
    --output_dir "$TRACES_DIR" \
    --workers 8

# Collect individual JSONs → flat JSONL
python -m reasonops.data.run_inference \
    --collect \
    --traces_dir "$TRACES_DIR" \
    --output     "$TRACES_DIR/../traces.jsonl.gz"
