#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/llm_judge_%j.out
#SBATCH --error=logs/llm_judge_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

# Each trace is judged by the same OpenRouter model that generated it.
# Requires OPENROUTER_API_KEY in environment or .env file.
${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.llm_judge \
    --corpus  "$DATA/final_dataset.jsonl.gz" \
    --output  "$DATA/predictions/preds_llm_judge.jsonl.gz" \
    --workers 4
