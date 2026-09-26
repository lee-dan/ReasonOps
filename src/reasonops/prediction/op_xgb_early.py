#!/usr/bin/env python3
"""
Op-XGB-Early: WP-AUC vs trace depth using full Op-XGB features.

For each depth d in {10, 25, 50, 75, 100}%, truncates each trace's spans to
the first ceil(N*d/100) spans, recomputes (i) the 117-dim op_features vector
on the truncated operator subsequence and (ii) anchor-phrase TF-IDF (8K
features, sublinear TF) on the truncated anchor text, concatenates them, and
fits XGBoost from scratch under problem-level 5-fold CV.

This matches the full Op-XGB feature recipe in op_seq_baseline.py.

Usage:
    python -m reasonops.prediction.op_xgb_early \\
        --corpus  data/final_dataset.jsonl.gz \\
        --output  data/predictions/preds_opxgb_early.jsonl.gz \\
        --summary data/predictions/opxgb_early_summary.json
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

from reasonops.utils import EXCLUDE_DATASETS, N_FOLDS, SEED, problem_kfold_splits, wp_auc, make_clf
from reasonops.prediction.op_seq_baseline import op_features

DEPTHS = [10, 25, 50, 75, 100]


def truncate_spans(spans, depth_pct):
    """Return the first ceil(N * depth_pct/100) spans of `spans`."""
    n = max(1, math.ceil(len(spans) * depth_pct / 100))
    return spans[:n]


def build_features_at_depth(all_spans, depth_pct):
    """Return (X_op, anchor_texts) computed on the depth-truncated traces.

    Operator sequence is derived from each visible span's `cluster` field
    (0--6 are operators; -1 indicates a structural/non-operator span and is
    excluded from the operator sequence, matching final_dataset.jsonl.gz).
    """
    seqs = []
    anchor_texts = []
    for spans in all_spans:
        partial = truncate_spans(spans, depth_pct)
        seq = []
        for sp in partial:
            c = sp.get("cluster")
            if c is None:
                continue
            try:
                ci = int(c)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < 7:
                seq.append(ci)
        seqs.append(seq)
        parts = [sp.get("anchor", "")[:80] for sp in partial if sp.get("anchor")]
        anchor_texts.append(" ".join(parts) if parts else "")
    X_op = np.array([op_features(s) for s in seqs], dtype=np.float32)
    return X_op, anchor_texts


def xgb_oof_combined(X_op, anchor_texts, y, pids, n_folds, seed):
    """Problem-level K-fold OOF with concatenated TF-IDF + op_features."""
    N = len(y)
    oof = np.full(N, np.nan)
    for fold, (tr, te) in enumerate(problem_kfold_splits(pids, n_folds, seed)):
        if len(set(y[tr].tolist())) < 2:
            continue
        tfidf = TfidfVectorizer(max_features=8000, sublinear_tf=True)
        T_tr = tfidf.fit_transform([anchor_texts[i] for i in tr]).toarray().astype(np.float32)
        T_te = tfidf.transform([anchor_texts[i] for i in te]).toarray().astype(np.float32)
        X_tr = np.hstack([T_tr, X_op[tr]])
        X_te = np.hstack([T_te, X_op[te]])
        clf = make_clf(seed)
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed",    type=int, default=SEED)
    args = ap.parse_args()

    print(f"Loading corpus: {args.corpus}", flush=True)
    opener = gzip.open if str(args.corpus).endswith(".gz") else open

    trace_ids, datasets, models, pids_list = [], [], [], []
    ys, all_spans = [], []

    with opener(args.corpus, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            ds = r.get("dataset", "")
            if ds in EXCLUDE_DATASETS:
                continue
            correct = r.get("correct")
            spans = r.get("spans", [])
            if correct is None or len(spans) == 0:
                continue
            trace_ids.append(r["trace_id"])
            datasets.append(ds)
            models.append(r.get("model", ""))
            pids_list.append(r.get("problem_id", ""))
            ys.append(int(bool(correct)))
            all_spans.append(spans)

    N = len(ys)
    y      = np.array(ys, dtype=np.int8)
    pids   = np.array(pids_list)
    ds_arr = np.array(datasets)
    print(f"  {N:,} traces, {len(np.unique(ds_arr))} datasets", flush=True)

    oof_by_depth = {}
    summary = {}

    for d in DEPTHS:
        print(f"\n[depth={d}%] building features (op + tfidf, combined)", flush=True)
        X_op, anchor_texts = build_features_at_depth(all_spans, d)
        print(f"  X_op shape={X_op.shape}, fitting CV…", flush=True)
        oof = xgb_oof_combined(X_op, anchor_texts, y, pids, args.n_folds, args.seed)
        oof_by_depth[d] = oof

        v = ~np.isnan(oof)
        global_wpauc = wp_auc(y[v], oof[v], pids[v])
        print(f"  global WP-AUC = {global_wpauc:.4f}", flush=True)

        per_ds = {}
        for ds in sorted(np.unique(ds_arr)):
            mask = ds_arr == ds
            vm = mask & v
            if vm.sum() < 5:
                continue
            auc = wp_auc(y[vm], oof[vm], pids[vm])
            per_ds[ds] = round(auc, 4)
            print(f"    {ds:30s} WP-AUC={auc:.4f}", flush=True)

        summary[f"d{d}"] = {"global": round(global_wpauc, 4), "per_dataset": per_ds}

    # Write predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opener_w = gzip.open if str(args.output).endswith(".gz") else open
    with opener_w(args.output, "wt") as out:
        for i in range(N):
            rec = {
                "trace_id":   trace_ids[i],
                "dataset":    datasets[i],
                "model":      models[i],
                "problem_id": pids_list[i],
                "correct":    int(y[i]),
            }
            for d in DEPTHS:
                v = oof_by_depth[d][i]
                rec[f"d{d}"] = None if np.isnan(v) else float(v)
            out.write(json.dumps(rec) + "\n")
    print(f"\nWrote {N:,} records → {args.output}", flush=True)

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary → {args.summary}", flush=True)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
