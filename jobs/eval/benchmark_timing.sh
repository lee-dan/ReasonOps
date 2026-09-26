#!/bin/bash
#SBATCH --job-name=benchmark_timing
#SBATCH --partition=normal
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/benchmark_timing_%j.out
#SBATCH --error=logs/benchmark_timing_%j.err

set -euo pipefail
source ~/.bashrc
source ${REASONOPS_VENV}/bin/activate
cd "$(dirname "$0")/../.."

if [ -f .env ]; then source .env; fi

VENV=${REASONOPS_VENV}

mkdir -p logs data

python -m reasonops.eval.benchmark_timing
