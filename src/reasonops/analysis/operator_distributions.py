#!/usr/bin/env python3
"""
Operator distribution diagnostics: per-model operator profiles and
leave-one-model-out (LOOMO) transfer analysis.

Outputs:
  fig_model_op_profiles.{pdf,png}  — z-score heatmap of operators × models
  fig_loomo_diagnostics.{pdf,png}  — best and worst LOOMO transfer models

Usage:
    python -m reasonops.analysis.operator_distributions \\
        --corpus     data/final_dataset.jsonl.gz \\
        --names      data/cluster_names.json \\
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

from reasonops.utils import MODEL_ORDER, MODEL_DISPLAY

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "figure.facecolor": "white",
})


def load_names(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v["name"] if isinstance(v, dict) else v for k, v in raw.items()}


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
                "y":       int(bool(correct)),
                "pid":     r.get("problem_id", ""),
                "dataset": r.get("dataset", ""),
                "model":   r.get("model", ""),
                "seq":     seq,
            }


def savefig(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def model_operator_profiles(rows, K, names, out_dir):
    per_model = defaultdict(list)
    for r in rows:
        seq = r["seq"]
        n = max(len(seq), 1)
        counts = np.zeros(K)
        for op in seq:
            if 0 <= op < K:
                counts[op] += 1
        per_model[r["model"]].append(counts / n)

    all_freqs = np.concatenate([np.array(v) for v in per_model.values()])
    global_mean = all_freqs.mean(0)
    global_std = all_freqs.std(0).clip(min=1e-8)

    ordered = [m for m in MODEL_ORDER if m in per_model]
    mat = np.zeros((len(ordered), K))
    for mi, m in enumerate(ordered):
        mf = np.mean(per_model[m], axis=0)
        mat[mi] = (mf - global_mean) / global_std

    op_labels = [names[i] for i in range(K)]
    model_labels = [MODEL_DISPLAY.get(m, m) for m in ordered]

    fig, ax = plt.subplots(figsize=(W, 3.2))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-2.0, vmax=2.0,
                   interpolation="nearest")
    ax.set_xticks(range(K))
    ax.set_xticklabels(op_labels, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(model_labels, fontsize=6.5)
    for mi in range(len(ordered)):
        for ki in range(K):
            v = mat[mi, ki]
            c = "white" if abs(v) > 1.0 else "#333"
            ax.text(ki, mi, f"{v:+.1f}", ha="center", va="center", fontsize=4.5, color=c)
    cb = fig.colorbar(im, ax=ax, label="Z-score vs corpus mean", fraction=0.03)
    cb.ax.tick_params(labelsize=5)
    ax.set_title("Per-model operator usage deviation (z-score)", fontsize=8)
    plt.tight_layout()
    savefig(fig, out_dir, "fig_model_op_profiles")

    print("\n  Per-model distinctive operators:")
    for mi, m in enumerate(ordered):
        hi = int(np.argmax(mat[mi]))
        lo = int(np.argmin(mat[mi]))
        print(f"    {MODEL_DISPLAY.get(m,m):<20}: "
              f"+{op_labels[hi]}({mat[mi,hi]:+.2f})  "
              f"-{op_labels[lo]}({mat[mi,lo]:+.2f})")

    return mat, ordered


def loomo_diagnostics(rows, K, names, out_dir):
    worst = {"claude-haiku", "claude-sonnet", "nemotron-49b"}
    best = {"qwq-32b", "grok-3-mini", "r1-distill-llama-70b"}

    per_model_freq = defaultdict(list)
    for r in rows:
        seq = r["seq"]
        n = len(seq)
        counts = np.zeros(K)
        for op in seq:
            if 0 <= op < K:
                counts[op] += 1
        per_model_freq[r["model"]].append(counts / n)

    per_model_mean = {m: np.mean(v, axis=0) for m, v in per_model_freq.items()}
    worst_models = [m for m in worst if m in per_model_mean]
    best_models  = [m for m in best  if m in per_model_mean]
    if not worst_models or not best_models:
        print("  Warning: LOOMO worst/best models not found in corpus — skipping")
        return per_model_mean

    mean_worst = np.mean([per_model_mean[m] for m in worst_models], axis=0)
    mean_best  = np.mean([per_model_mean[m] for m in best_models],  axis=0)

    op_labels = [names[i] for i in range(K)]
    x = np.arange(K)
    w = 0.35

    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.bar(x - w/2, mean_best, w, color="#117733", alpha=0.85,
           label="Best LOOMO (QwQ, Grok, R1-L70)")
    ax.bar(x + w/2, mean_worst, w, color="#882255", alpha=0.85,
           label="Worst LOOMO (Haiku, Sonnet, Nemotron)")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:8] for n in op_labels], fontsize=6.5)
    ax.set_ylabel("Mean operator frequency", fontsize=7)
    ax.legend(fontsize=6)
    plt.tight_layout()
    savefig(fig, out_dir, "fig_loomo_diagnostics")

    print("\n  Operator frequency: best vs worst LOOMO models")
    for i, name in enumerate(op_labels):
        delta = mean_worst[i] - mean_best[i]
        print(f"    {name:<16}: best={mean_best[i]:.3f}, worst={mean_worst[i]:.3f}, Δ={delta:+.3f}")

    return per_model_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",     required=True)
    ap.add_argument("--names",      required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = load_names(args.names)
    K = len(names)
    print(f"K={K}")

    print("Loading corpus...")
    rows = list(stream_corpus(args.corpus))
    print(f"  {len(rows):,} traces, {len(set(r['pid'] for r in rows)):,} problems")

    print("\n=== Model operator profiles ===")
    model_operator_profiles(rows, K, names, out)

    print("\n=== LOOMO diagnostics ===")
    loomo_diagnostics(rows, K, names, out)

    print(f"\nDone. Figures in {out}/")


if __name__ == "__main__":
    main()
