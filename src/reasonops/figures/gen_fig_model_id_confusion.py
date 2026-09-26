#!/usr/bin/env python3
"""
Generate fig_model_id_confusion.pdf from Op-XGB model identification predictions.

Renders a row-normalized 12-class confusion matrix as a heatmap with diagonal
cells boxed, near-zero off-diagonals left blank, and macro-averaged diagonal
recall in the title.

Usage:
    python -m reasonops.figures.gen_fig_model_id_confusion \
        --preds   data/model_id/model_id_opxgb_predictions.jsonl.gz \
        --summary data/model_id/model_id_opxgb_summary.json \
        --out     figures
"""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# Display order groups models by family: Claude → DeepSeek-R1 → Qwen-thinking → QwQ → Grok → Nemotron → Kimi
DISPLAY_ORDER = [
    "claude-sonnet", "claude-haiku",
    "r1-0528", "r1-distill-llama-70b", "r1-distill-qwen-32b",
    "qwen3-235b-thinking", "qwen3-30b-thinking", "qwq-32b",
    "grok-3-mini",
    "nemotron-49b",
    "kimi-k2", "kimi-k2.5",
]
DISPLAY_LABELS = {
    "claude-sonnet": "Claude Sonnet 4.5",
    "claude-haiku":  "Claude Haiku 4.5",
    "r1-0528":              "DeepSeek R1-0528",
    "r1-distill-llama-70b": "R1-Distill-Llama-70B",
    "r1-distill-qwen-32b":  "R1-Distill-Qwen-32B",
    "qwen3-235b-thinking":  "Qwen3-235B-Thinking",
    "qwen3-30b-thinking":   "Qwen3-30B-Thinking",
    "qwq-32b":              "QwQ-32B",
    "grok-3-mini":          "Grok-3-Mini",
    "nemotron-49b":         "Nemotron-Super-49B",
    "kimi-k2":              "Kimi-K2",
    "kimi-k2.5":            "Kimi-K2.5",
}


def fmt_cell(v: float) -> str:
    """Format like '.54' (no leading zero); blank for near-zero."""
    if v < 0.005:
        return ""
    s = f"{v:.2f}"
    return s.lstrip("0") if s.startswith("0.") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",   required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out",     default="figures")
    args = ap.parse_args()

    rows = []
    opener = gzip.open if str(args.preds).endswith(".gz") else open
    with opener(args.preds, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("model_id_pred") is not None:
                rows.append(r)
    print(f"Loaded {len(rows):,} predictions")

    idx_of = {m: i for i, m in enumerate(DISPLAY_ORDER)}
    n = len(DISPLAY_ORDER)

    cm = np.zeros((n, n), dtype=np.float64)
    for r in rows:
        true = r["model"]
        pred = r["model_id_pred"]
        if true in idx_of and pred in idx_of:
            cm[idx_of[true], idx_of[pred]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    diag = np.diag(cm_norm)
    macro_diag = float(diag.mean())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.6, 6.6))

    cmap = plt.cm.Purples
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1.0, aspect="equal")

    labels = [DISPLAY_LABELS[m] for m in DISPLAY_ORDER]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    # Cell text — leading-zero-stripped, blank for near-zero
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            txt = fmt_cell(val)
            if txt:
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.0, color=color)

    # Box diagonal cells in black for emphasis
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="black", linewidth=1.2))

    ax.set_xlabel("Predicted model", fontsize=10)
    ax.set_ylabel("True model", fontsize=10)
    ax.set_title(f"Model-ID confusion (row-normalized) — macro-avg diagonal = {macro_diag:.2f}",
                 fontsize=10.5, pad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$P(\mathrm{predicted}\,|\,\mathrm{true})$", fontsize=9)

    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_model_id_confusion.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
