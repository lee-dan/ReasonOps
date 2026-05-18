#!/bin/bash
#SBATCH --job-name=grade
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/grade_%j.out
#SBATCH --error=logs/grade_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.data.grade \
    --input   "$DATA/traces.jsonl.gz" \
    --output  "$DATA/traces_graded.jsonl.gz" \
    --workers 8
