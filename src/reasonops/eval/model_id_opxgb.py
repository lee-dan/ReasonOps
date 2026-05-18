#!/usr/bin/env python3
"""
Op-XGB adapted for 12-class model identification.

Uses the SAME feature recipe as Op-XGB for correctness prediction
(see reasonops.prediction.op_seq_baseline): a 117-dim handcrafted operator
feature vector concatenated with an 8,000-feature anchor-phrase TF-IDF
representation. XGBoost is trained as a multi-class classifier to predict the
source model under problem-level 5-fold CV.

Outputs:
  data/model_id/model_id_opxgb_predictions.jsonl.gz
    fields: trace_id, dataset, model, problem_id, correct,
            model_id_pred (str), model_id_prob (float, max prob),
            model_id_probs (dict model->prob)

  data/model_id/model_id_opxgb_summary.json
    fields: accuracy, macro_auc, per_model_auc, confusion_matrix

Usage:
    python -m reasonops.eval.model_id_opxgb \\
        --corpus data/final_dataset.jsonl.gz \\
        --output data/model_id/model_id_opxgb_predictions.jsonl.gz
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

# Reuse the canonical 117-dim op_features definition.
from reasonops.prediction.op_seq_baseline import op_features
from reasonops.utils import EXCLUDE_DATASETS, N_FOLDS, SEED, problem_kfold_splits, make_multiclass_clf

MAX_TFIDF_FEATURES = 8000


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    probs: np.ndarray, label_names: list[str]) -> dict:
    accuracy = float((y_true == y_pred).mean())

    aucs = {}
    for i, name in enumerate(label_names):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        aucs[name] = float(roc_auc_score(binary, probs[:, i]))
    macro_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")

    n = len(label_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_auc": round(macro_auc, 4),
        "per_model_auc": {k: round(v, 4) for k, v in aucs.items()},
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",   required=True)
    ap.add_argument("--output",   default="data/model_id/model_id_opxgb_predictions.jsonl.gz")
    ap.add_argument("--n_folds",  type=int, default=N_FOLDS)
    ap.add_argument("--seed",     type=int, default=SEED)
    args = ap.parse_args()

    print(f"Loading corpus: {args.corpus}", flush=True)
    opener = gzip.open if str(args.corpus).endswith(".gz") else open

    trace_ids, datasets, models_list, pids_list, corrects = [], [], [], [], []
    seqs, anchor_texts = [], []

    with opener(args.corpus, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            ds = r.get("dataset", "")
            if ds in EXCLUDE_DATASETS:
                continue
            model = r.get("model", "")
            if not model:
                continue
            seq = r.get("operator_sequence", [])
            if len(seq) == 0:
                continue
            trace_ids.append(r["trace_id"])
            datasets.append(ds)
            models_list.append(model)
            pids_list.append(r.get("problem_id", ""))
            corrects.append(int(bool(r.get("correct", 0))))
            seqs.append([int(x) for x in seq])
            anchor_parts = [sp.get("anchor", "")[:80]
                            for sp in r.get("spans", []) if sp.get("anchor")]
            resp = r.get("model_response") or r.get("reasoning") or ""
            anchor_texts.append(" ".join(anchor_parts) if anchor_parts else resp[-3000:])

    N = len(trace_ids)
    pids = np.array(pids_list)
    print(f"  {N:,} traces, {len(set(models_list))} models", flush=True)

    le = LabelEncoder()
    y = le.fit_transform(models_list)
    label_names = list(le.classes_)
    n_classes = len(label_names)
    print(f"  Classes: {label_names}", flush=True)

    X_op = np.array([op_features(s) for s in seqs], dtype=np.float32)
    print(f"  op-feature dim = {X_op.shape[1]}", flush=True)

    oof_pred  = np.full(N, -1, dtype=int)
    oof_probs = np.full((N, n_classes), np.nan)

    splits = problem_kfold_splits(pids, args.n_folds, args.seed)
    for fold, (tr, te) in enumerate(splits):
        print(f"\nFold {fold+1}/{args.n_folds}  (train={len(tr):,} test={len(te):,})", flush=True)

        tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, sublinear_tf=True)
        T_tr = tfidf.fit_transform([anchor_texts[i] for i in tr]).toarray().astype(np.float32)
        T_te = tfidf.transform([anchor_texts[i] for i in te]).toarray().astype(np.float32)
        X_tr = np.hstack([T_tr, X_op[tr]])
        X_te = np.hstack([T_te, X_op[te]])

        clf = make_multiclass_clf(n_classes, args.seed + fold)
        clf.fit(X_tr, y[tr])

        probs_te = clf.predict_proba(X_te)
        oof_probs[te] = probs_te
        oof_pred[te]  = probs_te.argmax(axis=1)

        fold_acc = (oof_pred[te] == y[te]).mean()
        fold_aucs = []
        for i in range(n_classes):
            b = (y[te] == i).astype(int)
            if b.sum() > 0 and b.sum() < len(b):
                fold_aucs.append(roc_auc_score(b, probs_te[:, i]))
        print(f"  acc={fold_acc:.4f}  macro-AUC={np.mean(fold_aucs):.4f}", flush=True)

    valid = oof_pred >= 0
    metrics = compute_metrics(y[valid], oof_pred[valid], oof_probs[valid], label_names)

    print(f"\n{'='*60}", flush=True)
    print(f"Op-XGB Model ID Results (op_features + TF-IDF)", flush=True)
    print(f"  Accuracy:   {metrics['accuracy']} (chance={1/n_classes:.3f})", flush=True)
    print(f"  Macro-AUC:  {metrics['macro_auc']}", flush=True)
    print(f"  Per-model AUC:", flush=True)
    for model, auc in sorted(metrics['per_model_auc'].items()):
        print(f"    {model:<35} {auc:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opener_w = gzip.open if str(args.output).endswith(".gz") else open
    with opener_w(args.output, "wt") as out:
        for i in range(N):
            probs_dict = {label_names[j]: round(float(oof_probs[i, j]), 6)
                          for j in range(n_classes)} if oof_pred[i] >= 0 else {}
            out.write(json.dumps({
                "trace_id":        trace_ids[i],
                "dataset":         datasets[i],
                "model":           models_list[i],
                "problem_id":      pids_list[i],
                "correct":         corrects[i],
                "model_id_pred":   label_names[oof_pred[i]] if oof_pred[i] >= 0 else None,
                "model_id_prob":   round(float(oof_probs[i].max()), 6) if oof_pred[i] >= 0 else None,
                "model_id_probs":  probs_dict,
            }) + "\n")
    print(f"Saved predictions → {args.output}", flush=True)

    summary_path = Path(args.output).parent / "model_id_opxgb_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved summary    → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
