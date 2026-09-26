#!/usr/bin/env python3
"""
Backtracking baseline: logistic regression on fraction of BACKTRACKING operators.

Produces two OOF probability columns using problem-level 5-fold CV:
  backtrack_within  — CV run separately inside each dataset
  backtrack_cross   — CV run across the full corpus

Usage:
    python -m reasonops.prediction.baseline_backtrack \
        --corpus data/final_dataset.jsonl.gz \
        --names  data/operators/cluster_names.json \
        --output data/predictions/preds_backtrack.jsonl.gz
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from reasonops.utils import EXCLUDE_DATASETS, N_FOLDS, SEED, problem_kfold_splits, wp_auc


def get_backtrack_id(names_path):
    with open(names_path) as f:
        names = json.load(f)
    for cid, entry in names.items():
        name = entry["name"] if isinstance(entry, dict) else str(entry)
        if "BACKTRACK" in name.upper():
            return int(cid)
    return None


def logreg_oof(X, y, pids, n_folds=N_FOLDS, seed=SEED):
    N = len(y)
    oof = np.full(N, np.nan)
    for tr, te in problem_kfold_splits(pids, n_folds, seed):
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  required=True)
    ap.add_argument("--names",   required=True, help="cluster_names.json")
    ap.add_argument("--output",  required=True)
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed",    type=int, default=SEED)
    args = ap.parse_args()

    backtrack_id = get_backtrack_id(args.names)
    print(f"BACKTRACKING operator id: {backtrack_id}", flush=True)

    opener_r = gzip.open if str(args.corpus).endswith(".gz") else open
    records = []
    with opener_r(args.corpus, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dataset", "") in EXCLUDE_DATASETS:
                continue
            correct = r.get("correct")
            if correct is None:
                continue
            seq = r.get("operator_sequence", [])
            if seq and backtrack_id is not None:
                n = max(len(seq), 1)
                feat = sum(1 for op in seq if op == backtrack_id) / n
            else:
                text = (r.get("reasoning") or r.get("model_response") or "").lower()
                words = text.split()
                feat = sum(1 for w in words if "backtrack" in w) / max(len(words), 1)
            records.append({
                "trace_id":   r["trace_id"],
                "dataset":    r.get("dataset", ""),
                "model":      r.get("model", ""),
                "problem_id": r.get("problem_id", ""),
                "correct":    int(bool(correct)),
                "_feat":      float(feat),
            })
    print(f"Loaded {len(records):,} traces", flush=True)

    y      = np.array([r["correct"] for r in records], dtype=np.int8)
    X      = np.array([r["_feat"]   for r in records], dtype=np.float32).reshape(-1, 1)
    pids   = np.array([r["problem_id"] for r in records])
    ds_arr = np.array([r["dataset"]    for r in records])

    # Cross: global problem-level 5-fold CV
    cross = logreg_oof(X, y, pids, args.n_folds, args.seed)
    v = ~np.isnan(cross)
    wauc = wp_auc(y[v], cross[v], pids[v])
    print(f"Cross  WP-AUC = {wauc:.4f}", flush=True)

    # Within: per-dataset problem-level 5-fold CV
    within = np.full(len(records), np.nan)
    for ds in np.unique(ds_arr):
        mask = ds_arr == ds
        idx  = np.where(mask)[0]
        if mask.sum() < 10 or len(set(y[mask].tolist())) < 2:
            continue
        sub = logreg_oof(X[mask], y[mask], pids[mask], args.n_folds, args.seed)
        within[idx] = sub
        sv = ~np.isnan(sub)
        if sv.sum() > 0:
            wds = wp_auc(y[mask][sv], sub[sv], pids[mask][sv])
            print(f"  {ds:<22} within WP-AUC={wds:.4f}  n={mask.sum()}", flush=True)

    vw = ~np.isnan(within)
    wauc_w = wp_auc(y[vw], within[vw], pids[vw])
    print(f"Within WP-AUC = {wauc_w:.4f}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opener_w = gzip.open if str(args.output).endswith(".gz") else open
    with opener_w(args.output, "wt") as out:
        for i, r in enumerate(records):
            out.write(json.dumps({
                "trace_id":          r["trace_id"],
                "dataset":           r["dataset"],
                "model":             r["model"],
                "problem_id":        r["problem_id"],
                "correct":           r["correct"],
                "backtrack_within":  None if np.isnan(within[i]) else float(within[i]),
                "backtrack_cross":   None if np.isnan(cross[i])  else float(cross[i]),
            }) + "\n")
    print(f"Wrote {len(records):,} records → {args.output}", flush=True)


if __name__ == "__main__":
    main()
