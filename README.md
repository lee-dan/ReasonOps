# ReasonOps

Code for **ReasonOps: Operator Segmentation for LLM Reasoning Traces**.

ReasonOps discovers discourse-level operators from chain-of-thought traces. The package includes operator discovery, validation, trace analysis, model identification, and correctness prediction.

## Setup

Requires Python 3.11 or later.

```bash
pip install -e .
export REASONOPS_DATA="$PWD/data"
```

To unpack the included trace archive:

```bash
cat data/traces.tar.gz.part-* | tar -xzf - -C data
```

Collection and LLM-based evaluation use `OPENROUTER_API_KEY` and/or `ANTHROPIC_API_KEY`. Existing traces can be analyzed without generating new ones.

## Usage

Modules run with `python -m reasonops.<module>`; use `--help` for arguments. The `jobs/` directory contains SLURM launchers for the main pipeline. Set `REASONOPS_VENV` to your environment directory when using them.

- `data/`: collection, grading, and filtering.
- `operators/`: pivot discovery, clustering, and trace annotation.
- `eval/`: judge validation, naming stability, sensitivity sweeps, and system-prompt experiments.
- `prediction/`: correctness predictors, early prediction, and test-time selection.
- `analysis/` and `figures/`: operator statistics and plots.
