#!/bin/bash
#SBATCH --job-name=llm_judge_partial
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/llm_judge_partial_%j.out
#SBATCH --error=logs/llm_judge_partial_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

DATA="${REASONOPS_DATA}"

# Partial-trace LLM judge: Claude Haiku reads first p% of raw reasoning text.
# Generates WP-AUC vs trace depth curve for AIME, GPQA, LiveCodeBench.
${REASONOPS_VENV}/bin/python3 -m reasonops.prediction.llm_judge_partial \
    --corpus   "$DATA/final_dataset.jsonl.gz" \
    --output   "$DATA/predictions/preds_llm_judge_partial.jsonl" \
    --datasets aime,gpqa,livecodebench \
    --workers  64
