#!/usr/bin/env python3
"""
Generate fig_dataset_heatmap_correctness.pdf: paired heatmaps (correct |
incorrect) of operator frequency per dataset, with a single colorbar.

Usage:
    python -m reasonops.figures.gen_fig_dataset_heatmap_correctness \\
        --corpus data/final_dataset.jsonl.gz \\
        --names  data/cluster_names.json \\
        --out    figures
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

DATASET_ORDER = ["aime", "math", "gpqa", "livecodebench",
                 "humaneval", "mmlu_pro", "arc_challenge", "bbh"]
DATASET_LABELS = {
    "aime": "AIME", "math": "MATH", "gpqa": "GPQA",
    "livecodebench": "LiveCode", "humaneval": "HumanEval",
    "mmlu_pro": "MMLU-Pro", "arc_challenge": "ARC-C", "bbh": "BBH",
}
OP_ORDER = ["INITIATING", "QUALIFYING", "GROUNDING", "INFERRING",
            "HYPOTHESIZING", "BACKTRACKING", "CONSTRAINING"]


def load_names(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): (v["name"] if isinstance(v, dict) else v).upper()
            for k, v in raw.items()}


def build_matrix(rows, names, K, datasets):
    freq = {c: defaultdict(lambda: np.zeros(K)) for c in (0, 1)}
    cnt  = {c: defaultdict(int) for c in (0, 1)}
    for r in rows:
        ds = r["dataset"]
        c  = r["y"]
        seq = r["seq"]
        n = len(seq)
        if n == 0 or ds not in datasets:
            continue
        counts = np.zeros(K)
        for op in seq:
            if 0 <= op < K:
                counts[op] += 1
        freq[c][ds] += counts / n
        cnt[c][ds]  += 1

    op_idx = {name: i for i, name in names.items()}

    def make_mat(c):
        mat = np.full((len(datasets), K), np.nan)
        for di, ds in enumerate(datasets):
            if cnt[c][ds] > 0:
                mat[di] = freq[c][ds] / cnt[c][ds] * 100
        # reorder columns to OP_ORDER
        col_map = [op_idx[op] for op in OP_ORDER if op in op_idx]
        return mat[:, col_map]

    return make_mat(1), make_mat(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--names",  required=True)
    ap.add_argument("--out",    default="figures")
    args = ap.parse_args()

    names = load_names(args.names)
    K = len(names)

    opener = gzip.open if args.corpus.endswith(".gz") else open
    rows = []
    with opener(args.corpus, "rt") as f:
        for line in f:
            r = json.loads(line)
            correct = r.get("correct")
            seq = r.get("operator_sequence", [])
            if correct is None or len(seq) == 0:
                continue
            rows.append({"y": int(bool(correct)), "dataset": r.get("dataset",""),
                         "seq": seq})
    print(f"Loaded {len(rows):,} traces")

    present = [ds for ds in DATASET_ORDER if any(r["dataset"] == ds for r in rows)]
    mat_c, mat_i = build_matrix(rows, names, K, present)
    ds_labels = [DATASET_LABELS.get(ds, ds) for ds in present]

    vmin = np.nanmin([mat_c, mat_i])
    vmax = np.nanmax([mat_c, mat_i])

    fig, (ax_c, ax_i) = plt.subplots(1, 2, figsize=(10, 3.2),
                                      gridspec_kw={"wspace": 0.08})
    cmap = "RdYlBu_r"

    for ax, mat, title in [(ax_c, mat_c, "Correct"), (ax_i, mat_i, "Incorrect")]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(OP_ORDER)))
        ax.set_xticklabels(OP_ORDER, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(ds_labels)))
        ax.set_yticklabels(ds_labels, fontsize=8)
        ax.set_title(title, fontsize=10, pad=6)
        for di in range(len(ds_labels)):
            for oi in range(len(OP_ORDER)):
                v = mat[di, oi]
                if not np.isnan(v):
                    c = "white" if v > (vmin + 0.6 * (vmax - vmin)) else "#222"
                    ax.text(oi, di, f"{v:.0f}", ha="center", va="center",
                            fontsize=6.5, color=c)

    # colorbar attached to right panel only — stays at far right
    cb = fig.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
    cb.set_label("Span frequency (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_i.set_yticklabels([])  # suppress y labels on right panel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_dataset_heatmap_correctness.{ext}",
                    bbox_inches="tight", dpi=180)
        print(f"Saved: {out}/fig_dataset_heatmap_correctness.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
