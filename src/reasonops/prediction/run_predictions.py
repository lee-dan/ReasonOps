#!/usr/bin/env python3
"""
Merge all pre-computed prediction files into one master dataset and print a
WP-AUC leaderboard.

Expected inputs in --preds_dir:
  preds_length.jsonl.gz     → length_within, length_cross
  preds_backtrack.jsonl.gz  → backtrack_within, backtrack_cross
  preds_wait_count.jsonl.gz → wait_within, wait_cross
  preds_op_seq.jsonl.gz     → op_seq_within, op_seq_cross,
                              op_seq_tfidf_within, op_seq_tfidf_cross
  preds_llm_judge.jsonl.gz  → llm_judge_prob

Expected inputs in --ost_dir (defaults to --preds_dir):
  preds_ost.jsonl.gz        → d10/25/50/75/100  (cross-dataset CV)
  preds_ost_within.jsonl.gz → d10/25/50/75/100  (within-dataset CV)

Usage:
    python -m reasonops.prediction.run_predictions \
        --corpus    data/final_dataset.jsonl.gz \
        --preds_dir data/predictions/ \
        --ost_dir   data/ost_cv/ \
        --output    data/predictions/all_predictions.jsonl.gz
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from reasonops.utils import EXCLUDE_DATASETS

# (filename, score_key, output_column, use_ost_dir)
PRED_FILES = [
    ("preds_length.jsonl.gz",     "length_within",       "length_within",       False),
    ("preds_length.jsonl.gz",     "length_cross",        "length_cross",        False),
    ("preds_backtrack.jsonl.gz",  "backtrack_within",    "backtrack_within",    False),
    ("preds_backtrack.jsonl.gz",  "backtrack_cross",     "backtrack_cross",     False),
    ("preds_wait_count.jsonl.gz", "wait_within",         "wait_within",         False),
    ("preds_wait_count.jsonl.gz", "wait_cross",          "wait_cross",          False),
    ("preds_op_seq.jsonl.gz",     "op_seq_within",       "op_seq_within",       False),
    ("preds_op_seq.jsonl.gz",     "op_seq_cross",        "op_seq_cross",        False),
    ("preds_op_seq.jsonl.gz",     "op_seq_tfidf_within", "op_xgb_within",       False),
    ("preds_op_seq.jsonl.gz",     "op_seq_tfidf_cross",  "op_xgb_cross",        False),
    ("preds_llm_judge.jsonl.gz",  "llm_judge_prob",      "llm_judge",           False),
    ("preds_ost.jsonl.gz",        "d10",                 "ost_d10_cross",       True),
    ("preds_ost.jsonl.gz",        "d25",                 "ost_d25_cross",       True),
    ("preds_ost.jsonl.gz",        "d50",                 "ost_d50_cross",       True),
    ("preds_ost.jsonl.gz",        "d75",                 "ost_d75_cross",       True),
    ("preds_ost.jsonl.gz",        "d100",                "ost_d100_cross",      True),
    ("preds_ost_within.jsonl.gz", "d10",                 "ost_d10_within",      True),
    ("preds_ost_within.jsonl.gz", "d25",                 "ost_d25_within",      True),
    ("preds_ost_within.jsonl.gz", "d50",                 "ost_d50_within",      True),
    ("preds_ost_within.jsonl.gz", "d75",                 "ost_d75_within",      True),
    ("preds_ost_within.jsonl.gz", "d100",                "ost_d100_within",     True),
]


def load_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = {}
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows[r["trace_id"]] = r
    return rows


def wp_auc(y, scores, pids):
    groups = defaultdict(list)
    for pid, yt, yp in zip(pids, y, scores):
        groups[pid].append((yt, yp))
    aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
            for v in groups.values() if len(set(p[0] for p in v)) == 2]
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",    required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--ost_dir",   default=None,
                    help="Directory for seq_pred outputs (default: same as preds_dir)")
    ap.add_argument("--output",    required=True)
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    ost_dir   = Path(args.ost_dir) if args.ost_dir else preds_dir

    print(f"Loading corpus: {args.corpus}", flush=True)
    opener_r = gzip.open if str(args.corpus).endswith(".gz") else open
    corpus = {}
    with opener_r(args.corpus, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dataset", "") in EXCLUDE_DATASETS:
                continue
            if r.get("correct") is None:
                continue
            corpus[r["trace_id"]] = {
                "trace_id":   r["trace_id"],
                "dataset":    r.get("dataset", ""),
                "model":      r.get("model", ""),
                "problem_id": r.get("problem_id", ""),
                "correct":    int(bool(r["correct"])),
            }
    print(f"  {len(corpus):,} labeled traces", flush=True)

    # Load prediction files (cache by filename to avoid re-reading)
    file_cache = {}
    pred_maps = []
    for fname, score_key, col, use_ost in PRED_FILES:
        fpath = (ost_dir if use_ost else preds_dir) / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found", flush=True)
            pred_maps.append(None)
            continue
        cache_key = str(fpath)
        if cache_key not in file_cache:
            file_cache[cache_key] = load_jsonl(fpath)
            print(f"  Loaded {fname}: {len(file_cache[cache_key]):,} rows", flush=True)
        pred_maps.append((file_cache[cache_key], score_key, col))

    # Merge into per-trace records
    records = []
    for tid, meta in corpus.items():
        rec = dict(meta)
        for entry in pred_maps:
            if entry is None:
                continue
            pmap, score_key, col = entry
            row = pmap.get(tid, {})
            rec[col] = row.get(score_key)
        records.append(rec)

    # WP-AUC leaderboard
    print(f"\n{'='*65}", flush=True)
    print(f"{'METHOD':<30} {'WP-AUC':>8}  {'N_PROBS':>8}", flush=True)
    print(f"{'='*65}", flush=True)

    y_all    = np.array([r["correct"]    for r in records])
    pids_all = np.array([r["problem_id"] for r in records])

    for _, _, col, _ in PRED_FILES:
        sc = np.array([r.get(col) if r.get(col) is not None else np.nan for r in records])
        valid = ~np.isnan(sc)
        if valid.sum() < 50 or len(set(y_all[valid].tolist())) < 2:
            print(f"  {col:<30} (no data)", flush=True)
            continue
        wauc, n_probs = wp_auc(y_all[valid], sc[valid], pids_all[valid])
        print(f"  {col:<30} {wauc:>8.4f}  {n_probs:>8}", flush=True)

    print(f"{'='*65}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opener_w = gzip.open if str(args.output).endswith(".gz") else open
    with opener_w(args.output, "wt") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records):,} records → {args.output}", flush=True)


if __name__ == "__main__":
    main()
