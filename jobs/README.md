# SLURM job scripts

One script per pipeline stage. Each script assumes:

- A Python venv at `./.venv/` (`uv venv && uv pip install -e .`)
- Raw traces extracted under `./data/traces/`
- Secrets in `./.env` (e.g. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`)

Submit from the repo root, e.g. `sbatch jobs/prediction/ost.sh`.

Stages (run in order):

```
data/         inference.sh → grade.sh → filter.sh
operators/    discover_operators.sh → assemble_dataset.sh
eval/         k_sweep.sh · judge_validation.sh · naming_stability.sh
              benchmark_timing.sh · model_id_opxgb.sh
prediction/   baseline_{length,backtrack,wait_count}.sh
              op_seq_baseline.sh · op_xgb_early.sh
              llm_judge.sh · llm_judge_partial.sh
              ost.sh · ost_within.sh · run_predictions.sh
analysis/     operator_distributions.sh · temporal_dynamics.sh
              transition_analysis.sh
figures/      figures.sh · gen_fig_early_pred.sh
```
