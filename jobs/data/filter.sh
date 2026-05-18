#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.data.filter \
    --input   "$DATA/traces_graded.jsonl.gz" \
    --output  "$DATA/traces_filtered.jsonl.gz"
