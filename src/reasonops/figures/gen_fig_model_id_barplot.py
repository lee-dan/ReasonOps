#!/usr/bin/env python3
"""
Generate one-vs-rest AUC bar plot from Op-XGB model identification predictions,
either at the per-family or per-model level.

Per-family scores aggregate each family's probability mass across its
constituent models before computing one-vs-rest AUC; per-model scores use
the raw per-model probabilities directly.

Usage:
    python -m reasonops.figures.gen_fig_model_id_barplot \\
        --preds data/model_id/model_id_opxgb_predictions.jsonl.gz \\
        --mode  family            # or: model
        --out   figures/model_id
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
from sklearn.metrics import roc_auc_score

from reasonops.utils import FAMILY, MODEL_DISPLAY_FULL, MODEL_ORDER


FAMILY_COLORS = {
    "Claude":      "#f1ba6f",   # tan / orange
    "DeepSeek R1": "#83b1d9",   # blue
    "Qwen / QwQ":  "#9ddca1",   # green
    "Grok":        "#bdbdbd",   # grey
    "Nemotron":    "#c8b6e2",   # lavender
    "Kimi":        "#f48a92",   # pink
}


def load_rows(path):
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("model_id_probs"):
                rows.append(r)
    return rows


def family_aucs(rows):
    families = sorted(set(FAMILY.values()))
    y_true, fam_probs = [], []
    for r in rows:
        fp = defaultdict(float)
        for m, p in r["model_id_probs"].items():
            fam = FAMILY.get(m)
            if fam:
                fp[fam] += float(p)
        y_true.append(FAMILY[r["model"]])
        fam_probs.append(dict(fp))

    aucs = {}
    for fam in families:
        binary = np.array([1 if y == fam else 0 for y in y_true])
        probs  = np.array([fp.get(fam, 0.0) for fp in fam_probs])
        if 0 < binary.sum() < len(binary):
            aucs[fam] = float(roc_auc_score(binary, probs))
    return aucs


def model_aucs(rows):
    models = sorted(FAMILY.keys())
    y_true = [r["model"] for r in rows]
    aucs = {}
    for m in models:
        binary = np.array([1 if y == m else 0 for y in y_true])
        probs  = np.array([float(r["model_id_probs"].get(m, 0.0)) for r in rows])
        if 0 < binary.sum() < len(binary):
            aucs[m] = float(roc_auc_score(binary, probs))
    return aucs


def render(aucs, *, mode, out_dir, x_lim, fig_size, label_fontsize):
    """Render a horizontal bar plot of `aucs`, sorted descending."""
    if mode == "family":
        labels  = {k: k for k in aucs}
        colors  = {k: FAMILY_COLORS.get(k, "#cccccc") for k in aucs}
        x_label = "Per-family AUC (one-vs-rest)"
        out_name = "auc_per_family"
        legend_title = "Family"
    else:
        labels  = MODEL_DISPLAY_FULL
        colors  = {k: FAMILY_COLORS[FAMILY[k]] for k in aucs}
        x_label = "Per-model AUC (one-vs-rest)"
        out_name = "auc_per_model"
        legend_title = "Family"

    sorted_keys = sorted(aucs, key=aucs.get, reverse=True)
    print(f"Per-{mode} AUC:")
    for k in sorted_keys:
        print(f"  {labels[k]:30s}  {aucs[k]:.4f}")

    fig, ax = plt.subplots(figsize=fig_size)
    y_pos = np.arange(len(sorted_keys))
    vals  = [aucs[k] for k in sorted_keys]
    cols  = [colors[k]   for k in sorted_keys]

    bars = ax.barh(y_pos, vals, color=cols, edgecolor="black", linewidth=0.6, height=0.7)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[k] for k in sorted_keys], fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xlim(*x_lim)
    ax.tick_params(axis="x", labelsize=9)

    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", xy=(v, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, annotation_clip=False)

    legend_families = sorted(set(FAMILY_COLORS.keys()))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], ec="black", lw=0.5)
                      for f in legend_families]
    ax.legend(legend_handles, legend_families, title=legend_title,
              loc="center left", bbox_to_anchor=(1.12, 0.5),
              fontsize=8.5, title_fontsize=9, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{out_name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180)
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--mode",  choices=["family", "model"], default="family")
    ap.add_argument("--out",   default="figures")
    args = ap.parse_args()

    rows = load_rows(args.preds)
    print(f"Loaded {len(rows):,} predictions")

    if args.mode == "family":
        aucs = family_aucs(rows)
        render(aucs, mode="family", out_dir=args.out,
               x_lim=(0.95, 1.005), fig_size=(5.8, 2.9), label_fontsize=9.5)
    else:
        aucs = model_aucs(rows)
        render(aucs, mode="model", out_dir=args.out,
               x_lim=(0.95, 1.005), fig_size=(6.4, 4.6), label_fontsize=8.5)


if __name__ == "__main__":
    main()
