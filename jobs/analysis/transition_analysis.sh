#!/bin/bash
#SBATCH --job-name=transition
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/transition_analysis_%j.out
#SBATCH --error=logs/transition_analysis_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

python -m reasonops.analysis.transition_analysis \
    --corpus     "$DATA/final_dataset.jsonl.gz" \
    --output_dir "$DATA/figures/"
