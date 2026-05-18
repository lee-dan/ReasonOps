# ReasonOps: Operator Discovery in LLM Chain-of-Thought

ReasonOps induces a compact vocabulary of seven discourse-level reasoning operators from 44,662 chain-of-thought traces across 12 thinking LLMs and 8 benchmarks, fully unsupervised, and uses them to identify the source model with near-perfect AUC and to predict answer correctness from partial reasoning traces.

## Headline numbers

| Metric | Value |
|---|---|
| Corpus | 44,662 traces · 12 LLMs · 8 benchmarks |
| Operators | K=7 (Cohen's κ = 0.693–0.720 across three LLM judges) |
| Model identification | Op-XGB macro-AUC = 0.987 · accuracy = 79.9% (chance: 8.3%) |
| Correctness (Op-XGB, cross-dataset) | WP-AUC = 0.701 · AIME: 0.838 ID |
| Correctness (OST, cross-dataset) | WP-AUC = 0.701 · AIME: 0.801 |
| Early prediction (OST, 50% trace) | WP-AUC = 0.664 |

## Method overview

| Stage | Summary |
|---|---|
| **Pivot extraction** | First 3 lowercase alphabetic tokens of each sentence. Keep pivots that appear in ≥100 traces across ≥3 datasets, with all tokens in the top-2,000 most frequent corpus tokens. Yields **5,464 accepted pivots**. |
| **Embedding** | Each pivot embedded with `intfloat/e5-small-v2` (384-dim, L2-normalized). |
| **Clustering** | KMeans with K ∈ {6..11}, 30 restarts. K chosen by maximizing Cohen's κ against an independent LLM judge. **K=7** wins. |
| **Operators** | INITIATING · QUALIFYING · GROUNDING · INFERRING · HYPOTHESIZING · BACKTRACKING · CONSTRAINING |
| **Annotation** | Per-span nearest-centroid lookup. End-to-end discovery + annotation runs in under 7 minutes on a single CPU core for the full corpus. |
| **Op-XGB** | XGBoost on a 117-dim handcrafted operator feature vector (frequencies, quartile localization, bigram transitions, run lengths, first/last one-hots, entropy/length scalars) concatenated with an 8,000-feature anchor-phrase TF-IDF representation. Used for both correctness prediction and 12-class model identification. |
| **OST** | A ~800K-parameter Transformer encoder over the discrete operator label sequence (4 layers, d=128, 4 heads, pre-LayerNorm). Trained with a pairwise contrastive loss within each problem; supports early prediction natively. |
| **Evaluation** | Within-problem AUC (WP-AUC), problem-level 5-fold cross-validation, both cross-dataset and within-dataset protocols. |

## Repository layout

```
src/reasonops/
├── utils.py        shared constants, display maps, and helpers (imported by all modules)
├── data/           run_inference.py · benchmarks.py · models.py · grade.py · filter.py
├── operators/      discover_operators.py · assemble_dataset.py
├── eval/           judge_validation.py · naming_stability.py · k_sweep.py
│                   model_id_opxgb.py · benchmark_timing.py · compute_summaries.py
├── prediction/     baseline_{length,backtrack,wait_count}.py · op_seq_baseline.py
│                   op_xgb_early.py · seq_pred.py · llm_judge.py · llm_judge_partial.py
│                   run_predictions.py
├── analysis/       operator_distributions.py · temporal_dynamics.py · transition_analysis.py
│                   scope_judge.py
└── figures/        gen_fig_dataset_heatmap_correctness.py · gen_fig_early_pred.py
                    gen_fig_model_id_confusion.py · gen_fig_model_id_barplot.py
                    gen_fig_stability.py

jobs/               SLURM scripts mirroring src/ sections
configs/            data.toml
paper/              main.tex · SI.tex · checklist.tex · neurips_2026.sty · figures/
data/               result files (gitignored): predictions, kappa, stability, model_id, timing
```

## Reproducing the pipeline

```bash
# 1. Install
uv pip install -e .

# 2. Collect traces (requires OPENROUTER_API_KEY and ANTHROPIC_API_KEY)
sbatch jobs/data/inference.sh
sbatch jobs/data/grade.sh
sbatch jobs/data/filter.sh

# 3. Discover operators and annotate the corpus
sbatch jobs/operators/discover_operators.sh
sbatch jobs/operators/assemble_dataset.sh

# 4. Evaluate operator stability
sbatch jobs/eval/k_sweep.sh              # K selection: K ∈ {6..11}
sbatch jobs/eval/judge_validation.sh     # three-judge κ on held-out spans
sbatch jobs/eval/naming_stability.sh
sbatch jobs/eval/benchmark_timing.sh

# 5. Correctness prediction
sbatch jobs/prediction/baseline_length.sh
sbatch jobs/prediction/baseline_backtrack.sh
sbatch jobs/prediction/baseline_wait_count.sh
sbatch jobs/prediction/op_seq_baseline.sh    # Op-XGB: full feature recipe
sbatch jobs/prediction/op_xgb_early.sh       # Op-XGB-Early: per-depth retrained
sbatch jobs/prediction/llm_judge.sh          # SelfCheck (full trace)
sbatch jobs/prediction/llm_judge_partial.sh  # SelfCheck (partial trace)
sbatch jobs/prediction/ost.sh                # OST cross-dataset (GPU)
sbatch jobs/prediction/ost_within.sh         # OST within-dataset (GPU)
sbatch jobs/prediction/run_predictions.sh    # merge all → WP-AUC leaderboard

# 6. Model identification
sbatch jobs/eval/model_id_opxgb.sh           # Op-XGB 12-class

# 7. Reproduce all paper numbers
python -m reasonops.eval.compute_summaries   # → data/wpauc_summary.json
```

## Required environment variables

```
OPENROUTER_API_KEY    # all non-Claude models
ANTHROPIC_API_KEY     # Claude Sonnet 4.5, Claude Haiku 4.5
```
