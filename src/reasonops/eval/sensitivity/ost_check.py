"""Evaluate OST prediction under alternative clusterings."""

from reasonops.paths import DATA_DIR, OUTPUT_DIR

HERE = OUTPUT_DIR / "sensitivity"
import json
import pickle
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import roc_auc_score

import reasonops.prediction.seq_pred as sp
from reasonops.utils import EXCLUDE_DATASETS

EMB = DATA_DIR / "k_sweep/ngram_embeddings.npy"
CACHE = HERE / "cache/trace_cache.pkl"
RES = HERE / "results"
SETTINGS = [
    ("k6", "kmeans", 6),
    ("k7", "kmeans", 7),
    ("k8", "kmeans", 8),
    ("agglom7", "agglomerative", 7),
]


def cluster(emb, algo, K, seed=42):
    if algo == "kmeans":
        return KMeans(n_clusters=K, random_state=seed, n_init=30, max_iter=500).fit_predict(emb)
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(emb)


def build_rows(cache, ng_labels):
    rows = []
    for tid, t in cache["traces"].items():
        if t["dataset"] in EXCLUDE_DATASETS or t["correct"] is None:
            continue
        seq = ng_labels[t["pivot_idx"]].tolist()
        if not seq:
            continue
        rows.append(
            {
                "trace_id": tid,
                "dataset": t["dataset"],
                "model": t["model"],
                "problem_id": t["problem_id"],
                "correct": int(bool(t["correct"])),
                "seq": seq,
            }
        )
    return rows


def global_wpauc(oof):
    from collections import defaultdict

    g = defaultdict(list)
    for r in oof:
        g[r["problem_id"]].append((r["correct"], r["d100"]))
    aucs = [
        roc_auc_score([x[0] for x in v], [x[1] for x in v])
        for v in g.values()
        if len(set((x[0] for x in v))) == 2
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    import torch

    (HERE / "cache").mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    emb = np.load(EMB)
    cache = pickle.loads(CACHE.read_bytes())
    args = SimpleNamespace(
        output_dir=str(HERE / "cache"),
        epochs=40,
        patience=10,
        batch_size=128,
        lr=0.0003,
        max_len=512,
        contrast_k=16,
        k_folds=5,
        seed=42,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    out = {}
    for tag, algo, K in SETTINGS:
        labels = cluster(emb, algo, K)
        Kc = int(labels.max()) + 1
        sp.N_OPS, sp.PAD_ID, sp.VOCAB = (Kc, Kc, Kc + 1)
        rows = build_rows(cache, labels)
        print(f"\n### {tag}: K={Kc}, {len(rows):,} traces", flush=True)
        oof = sp._cv_loop(args, rows)
        wp = global_wpauc(oof)
        out[tag] = {"K": Kc, "algo": algo, "ost_wpauc": round(wp, 4), "n_traces": len(rows)}
        print(f"{tag}: OST global WP-AUC = {wp:.4f}", flush=True)
    (RES / "ost_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
