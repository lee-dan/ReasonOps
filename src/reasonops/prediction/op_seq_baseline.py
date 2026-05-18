#!/usr/bin/env python3
"""
Operator-sequence XGBoost baseline (Op-XGB) for correctness prediction.

Op-XGB combines a 117-dim handcrafted operator feature vector with an
8,000-feature anchor-phrase TF-IDF representation; the concatenated feature
vector is fed to XGBoost. Both within-dataset (per-dataset CV) and
cross-dataset (pooled CV) are computed at problem-level 5-fold CV.

Outputs four OOF probability columns:
  op_seq_within       — op features only, CV within each dataset
  op_seq_cross        — op features only, CV across full corpus
  op_seq_tfidf_within — op features + TF-IDF, CV within each dataset (Op-XGB WI)
  op_seq_tfidf_cross  — op features + TF-IDF, CV across full corpus (Op-XGB CD)

Usage:
    python -m reasonops.prediction.op_seq_baseline \
        --corpus data/final_dataset.jsonl.gz \
        --output data/predictions/preds_op_seq.jsonl.gz
"""
import argparse
import gzip
import json
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

K = 7


def op_features(seq):
    """117-dim handcrafted operator-sequence feature vector.

    Layout:
      freq         (7)   — global operator frequencies
      quartile_*   (28)  — operator frequencies in each of 4 sequence quartiles
      scalars      (5)   — entropy, max-freq, num-distinct, log-length, repeat-rate
      first_oh     (7)   — one-hot of the first operator
      last_oh      (7)   — one-hot of the last operator
      bigrams      (49)  — bigram transition matrix flattened (normalized)
      run_mean     (7)   — mean operator run length (normalized by trace length)
      run_max      (7)   — max  operator run length (normalized by trace length)
    """
    n = len(seq)
    counts = np.zeros(K, dtype=np.float32)
    for op in seq:
        if 0 <= op < K:
            counts[op] += 1
    freq = counts / max(n, 1)

    q_size = max(1, n // 4)
    quartile_flat = []
    for q in range(4):
        s = q * q_size
        e = (q + 1) * q_size if q < 3 else n
        qc = np.zeros(K, dtype=np.float32)
        for op in seq[s:e]:
            if 0 <= op < K:
                qc[op] += 1
        quartile_flat.append(qc / max(e - s, 1))

    nz = freq[freq > 0]
    entropy = float(-np.sum(nz * np.log(nz + 1e-12))) if len(nz) else 0.0
    scalars = np.array([
        entropy,
        float(freq.max()) if n > 0 else 0.0,
        float((freq > 0).sum()),
        float(np.log1p(n)),
        float(sum(seq[i] == seq[i-1] for i in range(1, n))) / max(n - 1, 1),
    ], dtype=np.float32)

    first_oh = np.zeros(K, dtype=np.float32)
    last_oh  = np.zeros(K, dtype=np.float32)
    if n > 0 and 0 <= seq[0]  < K: first_oh[seq[0]]  = 1.0
    if n > 0 and 0 <= seq[-1] < K: last_oh[seq[-1]]  = 1.0

    bigrams = np.zeros(K * K, dtype=np.float32)
    for i in range(1, n):
        a, b = seq[i-1], seq[i]
        if 0 <= a < K and 0 <= b < K:
            bigrams[a * K + b] += 1
    if n > 1:
        bigrams /= (n - 1)

    run_mean = np.zeros(K, dtype=np.float32)
    run_max  = np.zeros(K, dtype=np.float32)
    runs = {k: [] for k in range(K)}
    if n > 0:
        cur, cur_len = seq[0], 1
        for op in seq[1:]:
            if op == cur:
                cur_len += 1
            else:
                if 0 <= cur < K: runs[cur].append(cur_len)
                cur, cur_len = op, 1
        if 0 <= cur < K: runs[cur].append(cur_len)
        for k in range(K):
            if runs[k]:
                run_mean[k] = float(np.mean(runs[k])) / max(n, 1)
                run_max[k]  = float(max(runs[k]))    / max(n, 1)

    return np.concatenate([freq, *quartile_flat, scalars, first_oh, last_oh,
                           bigrams, run_mean, run_max])


def xgb_oof(X_op, anchor_texts, y, pids, n_folds, seed, use_tfidf):
    """Problem-level K-fold OOF XGBoost.

    use_tfidf=False: 117-dim op features only.
    use_tfidf=True : 8K TF-IDF + 117 op features concatenated (full Op-XGB).
    """
    N = len(y)
    oof = np.full(N, np.nan)
    for fold, (tr, te) in enumerate(problem_kfold_splits(pids, n_folds, seed)):
        if len(set(y[tr].tolist())) < 2:
            continue
        if use_tfidf:
            tfidf = TfidfVectorizer(max_features=8000, sublinear_tf=True)
            T_tr = tfidf.fit_transform([anchor_texts[i] for i in tr]).toarray().astype(np.float32)
            T_te = tfidf.transform([anchor_texts[i] for i in te]).toarray().astype(np.float32)
            X_tr = np.hstack([T_tr, X_op[tr]])
            X_te = np.hstack([T_te, X_op[te]])
        else:
            X_tr, X_te = X_op[tr], X_op[te]
        clf = make_clf(seed)
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)[:, 1]
        tv = ~np.isnan(oof[te])
        if tv.sum() > 0:
            tag = "tfidf+op" if use_tfidf else "op_seq"
            print(f"  [{tag}] fold {fold+1}/{n_folds}: "
                  f"WP-AUC={wp_auc(y[te][tv], oof[te][tv], pids[te][tv]):.4f}",
                  flush=True)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed",    type=int, default=SEED)
    args = ap.parse_args()

    print(f"Loading corpus: {args.corpus}", flush=True)
    opener = gzip.open if str(args.corpus).endswith(".gz") else open

    trace_ids, datasets, models, pids_list = [], [], [], []
    ys, seqs, anchor_texts = [], [], []

    with opener(args.corpus, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            ds = r.get("dataset", "")
            if ds in EXCLUDE_DATASETS:
                continue
            correct = r.get("correct")
            seq = r.get("operator_sequence", [])
            if correct is None or len(seq) == 0:
                continue
            trace_ids.append(r["trace_id"])
            datasets.append(ds)
            models.append(r.get("model", ""))
            pids_list.append(r.get("problem_id", ""))
            ys.append(int(bool(correct)))
            seqs.append([int(x) for x in seq])
            anchor_parts = [sp.get("anchor", "")[:80]
                            for sp in r.get("spans", []) if sp.get("anchor")]
            resp = r.get("model_response") or r.get("reasoning") or ""
            anchor_texts.append(" ".join(anchor_parts) if anchor_parts else resp[-3000:])

    N = len(ys)
    y      = np.array(ys, dtype=np.int8)
    pids   = np.array(pids_list)
    ds_arr = np.array(datasets)
    X_op   = np.array([op_features(s) for s in seqs], dtype=np.float32)
    print(f"  {N:,} traces, op-feature dim = {X_op.shape[1]}", flush=True)

    print("\n[cross / op_seq only]", flush=True)
    op_cross    = xgb_oof(X_op, anchor_texts, y, pids, args.n_folds, args.seed, use_tfidf=False)
    print("\n[cross / op_seq + tfidf]", flush=True)
    tfidf_cross = xgb_oof(X_op, anchor_texts, y, pids, args.n_folds, args.seed, use_tfidf=True)

    v = ~np.isnan(op_cross)
    print(f"\nCross op_seq        WP-AUC = {wp_auc(y[v], op_cross[v], pids[v]):.4f}", flush=True)
    v = ~np.isnan(tfidf_cross)
    print(f"Cross op_seq+tfidf  WP-AUC = {wp_auc(y[v], tfidf_cross[v], pids[v]):.4f}", flush=True)

    op_within    = np.full(N, np.nan)
    tfidf_within = np.full(N, np.nan)
    for ds in np.unique(ds_arr):
        mask = ds_arr == ds
        idx  = np.where(mask)[0]
        if mask.sum() < 20 or len(set(y[mask].tolist())) < 2:
            continue
        print(f"\n[within / {ds}]", flush=True)
        sub_op    = xgb_oof(X_op[mask], [anchor_texts[i] for i in idx],
                            y[mask], pids[mask], args.n_folds, args.seed, use_tfidf=False)
        sub_tfidf = xgb_oof(X_op[mask], [anchor_texts[i] for i in idx],
                            y[mask], pids[mask], args.n_folds, args.seed, use_tfidf=True)
        op_within[idx]    = sub_op
        tfidf_within[idx] = sub_tfidf
        sv = ~np.isnan(sub_op)
        if sv.sum() > 0:
            print(f"  op_seq        WP-AUC={wp_auc(y[mask][sv], sub_op[sv], pids[mask][sv]):.4f}",
                  flush=True)
            print(f"  op_seq+tfidf  WP-AUC={wp_auc(y[mask][sv], sub_tfidf[sv], pids[mask][sv]):.4f}",
                  flush=True)

    v = ~np.isnan(op_within)
    print(f"\nWithin op_seq        WP-AUC = {wp_auc(y[v], op_within[v], pids[v]):.4f}", flush=True)
    v = ~np.isnan(tfidf_within)
    print(f"Within op_seq+tfidf  WP-AUC = {wp_auc(y[v], tfidf_within[v], pids[v]):.4f}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opener_w = gzip.open if str(args.output).endswith(".gz") else open
    with opener_w(args.output, "wt") as out:
        for i in range(N):
            out.write(json.dumps({
                "trace_id":            trace_ids[i],
                "dataset":             datasets[i],
                "model":               models[i],
                "problem_id":          pids_list[i],
                "correct":             int(y[i]),
                "op_seq_within":       None if np.isnan(op_within[i])    else float(op_within[i]),
                "op_seq_cross":        None if np.isnan(op_cross[i])     else float(op_cross[i]),
                "op_seq_tfidf_within": None if np.isnan(tfidf_within[i]) else float(tfidf_within[i]),
                "op_seq_tfidf_cross":  None if np.isnan(tfidf_cross[i])  else float(tfidf_cross[i]),
            }) + "\n")
    print(f"\nWrote {N:,} records → {args.output}", flush=True)


if __name__ == "__main__":
    main()
