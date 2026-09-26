#!/usr/bin/env python3
"""
Temporal dynamics diagnostics: how operator usage varies across trace depth
and a per-model × per-dataset WP-AUC heatmap.

Outputs:
  fig_temporal_xgb_early.{pdf,png}     — XGBoost WP-AUC at 10/25/50/75/100% depth
  fig_per_model_per_dataset.{pdf,png}  — heatmap of WP-AUC by model × dataset

Usage:
    python -m reasonops.analysis.temporal_dynamics \\
        --corpus     data/final_dataset.jsonl.gz \\
        --output_dir out/analysis/
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "figure.facecolor": "white",
})

from reasonops.utils import MODEL_DISPLAY, DS_DISPLAY, FRACS

W = 5.5


def stream_corpus(corpus_path):
    opener = gzip.open if str(corpus_path).endswith(".gz") else open
    with opener(corpus_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            correct = r.get("correct")
            seq = r.get("operator_sequence", [])
            if correct is None or len(seq) == 0:
                continue
            yield {
                "y": int(bool(correct)),
                "pid": r.get("problem_id", ""),
                "dataset": r.get("dataset", ""),
                "model": r.get("model", ""),
                "seq": seq,
            }


def op_features_at_depth(seq, K, depth):
    n_full = len(seq)
    n = max(1, int(n_full * depth))
    seq = seq[:n]
    counts = np.zeros(K)
    for op in seq:
        if 0 <= op < K:
            counts[op] += 1
    freq = counts / n
    q_size = max(1, n // 4)
    quartile = []
    for q in range(4):
        s, e = q * q_size, (q + 1) * q_size if q < 3 else n
        qc = np.zeros(K)
        for op in seq[s:e]:
            if 0 <= op < K:
                qc[op] += 1
        quartile.extend(qc / max(e - s, 1))
    nz = freq[freq > 0]
    entropy = float(-np.sum(nz * np.log(nz))) if len(nz) else 0.0
    self_loops = sum(seq[i] == seq[i-1] for i in range(1, n)) / max(n-1, 1)
    first_oh = np.zeros(K); last_oh = np.zeros(K)
    if 0 <= seq[0] < K: first_oh[seq[0]] = 1.0
    if 0 <= seq[-1] < K: last_oh[seq[-1]] = 1.0
    bigrams = np.zeros(K * K)
    for i in range(1, n):
        a, b = seq[i-1], seq[i]
        if 0 <= a < K and 0 <= b < K:
            bigrams[a * K + b] += 1
    if n > 1:
        bigrams /= (n - 1)
    run_mean = np.zeros(K); run_max = np.zeros(K)
    if n > 0:
        runs = {k: [] for k in range(K)}
        cur, cur_len = seq[0], 1
        for op in seq[1:]:
            if op == cur:
                cur_len += 1
            else:
                if 0 <= cur < K:
                    runs[cur].append(cur_len)
                cur, cur_len = op, 1
        if 0 <= cur < K:
            runs[cur].append(cur_len)
        for k in range(K):
            if runs[k]:
                run_mean[k] = np.mean(runs[k]) / max(n, 1)
                run_max[k] = max(runs[k]) / max(n, 1)
    op_full = np.concatenate([
        freq, np.array(quartile),
        [entropy, float(freq.max()), float((freq > 0).sum()),
         float(np.log1p(n)), float(self_loops)],
        first_oh, last_oh,
    ])
    return np.concatenate([op_full, bigrams, run_mean, run_max])


def wp_auc(y_true, y_pred, problem_ids):
    groups = defaultdict(list)
    for pid, yt, yp in zip(problem_ids, y_true, y_pred):
        groups[pid].append((yt, yp))
    aucs = []
    for pairs in groups.values():
        yt = [p[0] for p in pairs]; yp = [p[1] for p in pairs]
        if len(set(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def problem_level_cv(X, y, pids, k=5, seed=42):
    unique_pids = np.array(sorted(set(pids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)
    pid_folds = np.array_split(unique_pids, k)
    wp_aucs = []
    for fold_idx in range(k):
        te_pids = set(pid_folds[fold_idx])
        tr_pids = set(unique_pids) - te_pids
        tr = np.where([p in tr_pids for p in pids])[0]
        te = np.where([p in te_pids for p in pids])[0]
        if len(set(y[tr])) < 2 or len(te) == 0:
            continue
        clf = (XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=seed,
                             n_jobs=-1, verbosity=0) if HAS_XGB else
               RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=seed))
        clf.fit(X[tr], y[tr])
        preds = clf.predict_proba(X[te])[:, 1]
        wauc, _ = wp_auc(y[te], preds, pids[te])
        wp_aucs.append(wauc)
    return float(np.mean(wp_aucs)) if wp_aucs else float("nan"), float(np.std(wp_aucs)) if wp_aucs else 0.0


def savefig(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def early_prediction(rows, K, k_folds, out_dir):
    depths = [0.1, 0.25, 0.5, 0.75, 1.0]
    y = np.array([r["y"] for r in rows])
    pids = np.array([r["pid"] for r in rows])
    seqs = [r["seq"] for r in rows]

    results = {}
    for d in depths:
        print(f"  depth={d:.2f} ...", end=" ", flush=True)
        X = np.array([op_features_at_depth(s, K, d) for s in seqs])
        mu, sd = problem_level_cv(X, y, pids, k=k_folds)
        results[d] = (mu, sd)
        print(f"WP-AUC={mu:.4f} ± {sd:.4f}")

    ds = sorted(results.keys())
    mus = [results[d][0] for d in ds]
    sds = [results[d][1] for d in ds]
    fig, ax = plt.subplots(figsize=(W * 0.7, 2.2))
    ax.plot([d * 100 for d in ds], mus, "o-", color="#117733", lw=1.5, ms=5)
    ax.fill_between([d * 100 for d in ds],
                    [m - s for m, s in zip(mus, sds)],
                    [m + s for m, s in zip(mus, sds)],
                    alpha=0.15, color="#117733")
    ax.axhline(0.5, ls=":", c="#999", lw=0.6)
    ax.set_xlabel("Trace depth (%)", fontsize=7)
    ax.set_ylabel("WP-AUC", fontsize=7)
    ax.set_xticks([d * 100 for d in ds])
    ax.set_ylim(0.48, min(1.0, max(mus) + 0.06))
    for d, mu in zip(ds, mus):
        ax.text(d * 100, mu + 0.005, f"{mu:.3f}", ha="center", fontsize=6)
    plt.tight_layout()
    savefig(fig, out_dir, "fig_temporal_xgb_early")
    return results


def per_model_per_dataset_heatmap(rows, K, out_dir):
    y_all = np.array([r["y"] for r in rows])
    pids_all = np.array([r["pid"] for r in rows])
    models_all = np.array([r["model"] for r in rows])
    datasets_all = np.array([r["dataset"] for r in rows])
    X_all = np.array([op_features_at_depth(r["seq"], K, 1.0) for r in rows])

    unique_models = sorted(set(models_all))
    ds_order = ["aime", "math", "gpqa", "livecodebench",
                "humaneval", "mmlu_pro", "arc_challenge", "bbh"]
    unique_datasets = [d for d in ds_order if d in set(datasets_all)]

    heatmap = np.full((len(unique_models), len(unique_datasets)), np.nan)
    for mi, model in enumerate(unique_models):
        for di, dataset in enumerate(unique_datasets):
            mask = (models_all == model) & (datasets_all == dataset)
            if mask.sum() < 20:
                continue
            y_sub = y_all[mask]; pid_sub = pids_all[mask]; X_sub = X_all[mask]
            if len(set(y_sub)) < 2 or len(set(pid_sub)) < 4:
                continue
            k_sub = min(5, max(2, len(set(pid_sub)) // 4))
            mu, _ = problem_level_cv(X_sub, y_sub, pid_sub, k=k_sub)
            heatmap[mi, di] = mu

    model_labels = [MODEL_DISPLAY.get(m, m) for m in unique_models]
    ds_labels = [DS_DISPLAY.get(d, d) for d in unique_datasets]

    fig, ax = plt.subplots(figsize=(W, 3.2))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(unique_datasets)))
    ax.set_xticklabels(ds_labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(unique_models)))
    ax.set_yticklabels(model_labels, fontsize=6.5)
    plt.colorbar(im, ax=ax, label="WP-AUC", fraction=0.03)
    for mi in range(len(unique_models)):
        for di in range(len(unique_datasets)):
            v = heatmap[mi, di]
            if not np.isnan(v):
                ax.text(di, mi, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if 0.45 < v < 0.85 else "white")
    ax.set_title("Op-seq WP-AUC per model × dataset", fontsize=8)
    plt.tight_layout()
    savefig(fig, out_dir, "fig_per_model_per_dataset")

    print("\n  Per-model mean WP-AUC:")
    for mi, model in enumerate(unique_models):
        row_vals = heatmap[mi, ~np.isnan(heatmap[mi])]
        if len(row_vals) > 0:
            print(f"    {MODEL_DISPLAY.get(model, model):<20}: {np.mean(row_vals):.4f}")

    return heatmap, unique_models, unique_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",     required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k_folds",    type=int, default=5)
    ap.add_argument("--k_operators",type=int, default=7)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    K = args.k_operators

    print("Loading corpus...")
    rows = list(stream_corpus(args.corpus))
    print(f"  {len(rows):,} traces, {len(set(r['pid'] for r in rows)):,} problems")

    print("\n=== Early prediction (XGBoost depth sweep) ===")
    early_prediction(rows, K, args.k_folds, out)

    print("\n=== Per-model × per-dataset heatmap ===")
    per_model_per_dataset_heatmap(rows, K, out)

    print(f"\nDone. Figures in {out}/")


if __name__ == "__main__":
    main()
