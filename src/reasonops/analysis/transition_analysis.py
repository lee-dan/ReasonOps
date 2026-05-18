#!/usr/bin/env python3
"""
Transition diagnostics: dataset-level statistics and per-dataset model
specialization.

Outputs:
  fig_dataset_comparison.{pdf,png}  — WP-AUC vs difficulty/length/coverage per dataset
  fig_per_dataset_fit.{pdf,png}     — in-dataset vs global model specialization

Usage:
    python -m reasonops.analysis.transition_analysis \\
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

from reasonops.utils import DS_DISPLAY

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


def op_features(seq, K):
    n = max(len(seq), 1)
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
    return np.concatenate([freq, np.array(quartile),
                           [entropy, float(freq.max()), float((freq > 0).sum()),
                            float(np.log1p(n)), float(self_loops)],
                           first_oh, last_oh, bigrams, run_mean, run_max])


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


def dataset_comparison(rows, K, out_dir):
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    ds_results = {}
    for ds, ds_rows in sorted(by_ds.items()):
        n = len(ds_rows)
        correct_rate = float(np.mean([r["y"] for r in ds_rows]))
        lens = [len(r["seq"]) for r in ds_rows]
        n_pids = len(set(r["pid"] for r in ds_rows))
        y_s = np.array([r["y"] for r in ds_rows])
        pids_s = np.array([r["pid"] for r in ds_rows])
        X_s = np.array([op_features(r["seq"], K) for r in ds_rows])
        if len(set(y_s)) >= 2 and len(set(pids_s)) >= 4:
            k_s = min(5, max(2, len(set(pids_s)) // 4))
            wp_mu, wp_sd = problem_level_cv(X_s, y_s, pids_s, k=k_s)
        else:
            wp_mu, wp_sd = float("nan"), 0.0
        ds_results[ds] = {
            "n": n, "correct_rate": correct_rate,
            "median_len": float(np.median(lens)),
            "n_pids": n_pids,
            "traces_per_pid": n / max(n_pids, 1),
            "wp_auc": wp_mu, "wp_std": wp_sd,
        }
        print(f"  {ds:<16}: n={n:>5}, correct={correct_rate:.2%}, "
              f"med_len={np.median(lens):.0f}, n_probs={n_pids}, WP-AUC={wp_mu:.4f}")

    ds_ord = ["aime", "math", "gpqa", "livecodebench", "humaneval", "mmlu_pro", "arc_challenge", "bbh"]
    ds_plot = [d for d in ds_ord if d in ds_results and not np.isnan(ds_results[d]["wp_auc"])]
    colors_ds = {
        "aime": "#882255", "math": "#CC6677", "gpqa": "#332288",
        "livecodebench": "#44AA99", "humaneval": "#117733",
        "mmlu_pro": "#DDCC77", "arc_challenge": "#88CCEE", "bbh": "#AA4499",
    }

    fig, axes = plt.subplots(1, 3, figsize=(W * 1.4, 2.2))
    for ds in ds_plot:
        s = ds_results[ds]
        c = colors_ds.get(ds, "#555")
        lbl = DS_DISPLAY.get(ds, ds)
        for ax, xval, xlabel in [
            (axes[0], s["correct_rate"] * 100, "Accuracy (%)"),
            (axes[1], s["median_len"],          "Median trace length (spans)"),
            (axes[2], s["traces_per_pid"],       "Traces per problem"),
        ]:
            ax.scatter(xval, s["wp_auc"], color=c, s=45, zorder=3)
            ax.annotate(lbl, (xval, s["wp_auc"]), fontsize=5.5,
                        xytext=(4, 2), textcoords="offset points")

    for ax, title, xlabel in zip(
        axes,
        ["WP-AUC vs difficulty", "WP-AUC vs trace length", "WP-AUC vs coverage"],
        ["Accuracy (%)", "Median trace length (spans)", "Traces per problem"],
    ):
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("WP-AUC", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.axhline(0.5, ls=":", c="#999", lw=0.6)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    savefig(fig, out_dir, "fig_dataset_comparison")
    return ds_results


def per_dataset_fit(rows, K, out_dir):
    y_all = np.array([r["y"] for r in rows])
    pids_all = np.array([r["pid"] for r in rows])
    X_all = np.array([op_features(r["seq"], K) for r in rows])
    global_mu, global_sd = problem_level_cv(X_all, y_all, pids_all, k=5)
    print(f"  Global WP-AUC: {global_mu:.4f} ± {global_sd:.4f}")

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    ds_ord = ["aime", "math", "gpqa", "livecodebench", "humaneval", "mmlu_pro", "arc_challenge", "bbh"]
    per_ds_mu, per_ds_sd = {}, {}
    for ds in ds_ord:
        if ds not in by_ds or len(by_ds[ds]) < 40:
            continue
        ds_rows = by_ds[ds]
        y_s = np.array([r["y"] for r in ds_rows])
        p_s = np.array([r["pid"] for r in ds_rows])
        X_s = np.array([op_features(r["seq"], K) for r in ds_rows])
        if len(set(y_s)) < 2 or len(set(p_s)) < 4:
            continue
        k_s = min(5, max(2, len(set(p_s)) // 4))
        mu, sd = problem_level_cv(X_s, y_s, p_s, k=k_s)
        per_ds_mu[ds] = mu; per_ds_sd[ds] = sd
        print(f"  {ds:<16}: WP-AUC={mu:.4f} ± {sd:.4f}")

    ds_plot = [d for d in ds_ord if d in per_ds_mu]
    x = np.arange(len(ds_plot))
    mus = [per_ds_mu[d] for d in ds_plot]
    sds = [per_ds_sd[d] for d in ds_plot]

    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.bar(x, mus, 0.6, color="#332288", alpha=0.85, label="Per-dataset model")
    ax.errorbar(x, mus, yerr=sds, fmt="none", c="black", capsize=2.5, lw=0.9)
    ax.axhline(global_mu, ls="--", c="#882255", lw=1.1,
               label=f"Global model ({global_mu:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_DISPLAY.get(d, d) for d in ds_plot],
                       rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("WP-AUC", fontsize=7)
    ax.set_title("In-dataset vs global model specialization", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_ylim(0.4, min(1.0, max(mus) + 0.12))
    for xi, (m, s) in enumerate(zip(mus, sds)):
        ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=5.5)
    plt.tight_layout()
    savefig(fig, out_dir, "fig_per_dataset_fit")

    return {"global": global_mu, "per_dataset": per_ds_mu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",      required=True)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--k_operators", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    K = args.k_operators

    print("Loading corpus...")
    rows = list(stream_corpus(args.corpus))
    print(f"  {len(rows):,} traces, {len(set(r['pid'] for r in rows)):,} problems")

    print("\n=== Dataset comparison ===")
    dataset_comparison(rows, K, out)

    print("\n=== Per-dataset model fitting ===")
    per_dataset_fit(rows, K, out)

    print(f"\nDone. Figures in {out}/")


if __name__ == "__main__":
    main()
