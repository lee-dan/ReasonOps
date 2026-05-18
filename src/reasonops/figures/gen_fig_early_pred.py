#!/usr/bin/env python3
"""
Generate fig_early_prediction.pdf: OST WP-AUC vs trace depth, with the
Op-XGB-Early upper bound (TF-IDF + 117-dim operator features, retrained per
depth) and the partial-trace SelfCheck baseline overlaid for comparison.

Usage:
    python -m reasonops.figures.gen_fig_early_pred \\
        --oof     data/ost_cv/preds_ost.jsonl.gz \\
        --opxgb   data/predictions/preds_opxgb_early.jsonl.gz \\
        --llm_dir data/predictions \\
        --out     figures
"""
import argparse
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

FRACS = [10, 25, 50, 75, 100]
DATASETS = ["aime", "livecodebench", "gpqa"]
DS_LABELS = {"aime": "AIME", "livecodebench": "LiveCodeBench", "gpqa": "GPQA"}
TITLE_COLORS = {"aime": "#1e3a5c", "livecodebench": "#2d6a4f", "gpqa": "#7b2d8b"}
OST_COLOR  = "#1e3a5c"  # navy — single consistent OST color across panels
OPXGB_COLOR = "#d97706"  # amber — text-feature upper bound retrained per depth
LLM_COLOR  = "#c0392b"   # red — SelfCheck baseline


def wp_auc_and_ci(rows_ds, frac_key, n_boot=2000, seed=42):
    groups = defaultdict(list)
    for r in rows_ds:
        v = r.get(frac_key)
        if v is not None:
            groups[r["problem_id"]].append((r["correct"], v))
    per_prob = np.array([
        roc_auc_score([x[0] for x in v], [x[1] for x in v])
        for v in groups.values() if len(set(x[0] for x in v)) == 2
    ], dtype=np.float64)
    if len(per_prob) == 0:
        return np.nan, np.nan, np.nan
    point = float(per_prob.mean())
    rng = np.random.default_rng(seed)
    boot = rng.choice(per_prob, size=(n_boot, len(per_prob)), replace=True).mean(axis=1)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_llm_partial(llm_dir):
    rows = {}
    # old monolithic file
    old = Path(llm_dir) / "preds_llm_judge_partial.jsonl"
    if old.exists():
        for r in load_jsonl(old):
            if all(f"d{p}" in r for p in FRACS):
                rows[r["trace_id"]] = r
    # per-model shard files
    for fp in glob.glob(str(Path(llm_dir) / "preds_llm_judge_partial_*.jsonl")):
        for r in load_jsonl(fp):
            if all(f"d{p}" in r for p in FRACS):
                rows[r["trace_id"]] = r
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",     default="data/ost_cv/preds_ost.jsonl.gz", help="OST OOF JSONL(.gz)")
    ap.add_argument("--llm_dir", default="",    help="dir containing preds_llm_judge_partial*.jsonl")
    ap.add_argument("--opxgb",   default="",    help="Op-XGB-Early predictions JSONL(.gz)")
    ap.add_argument("--out",     default="figures")
    args = ap.parse_args()

    ost_rows = load_jsonl(args.oof)
    print(f"OST: {len(ost_rows):,} rows")

    llm_rows = []
    if args.llm_dir:
        llm_rows = load_llm_partial(args.llm_dir)
        print(f"LLM partial: {len(llm_rows):,} rows")

    opxgb_rows = []
    if args.opxgb:
        opxgb_rows = load_jsonl(args.opxgb)
        print(f"Op-XGB-Early: {len(opxgb_rows):,} rows")

    ost_by_ds  = defaultdict(list)
    for r in ost_rows:
        ost_by_ds[r["dataset"]].append(r)

    llm_by_ds = defaultdict(list)
    for r in llm_rows:
        llm_by_ds[r["dataset"]].append(r)

    opxgb_by_ds = defaultdict(list)
    for r in opxgb_rows:
        opxgb_by_ds[r["dataset"]].append(r)

    x = np.array(FRACS)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=False)
    fig.subplots_adjust(wspace=0.40, left=0.065, right=0.95, top=0.88, bottom=0.17)

    for ax, ds in zip(axes, DATASETS):
        # Compute all three curves first so we can position labels without overlap
        # OST
        ost_aucs, ost_los, ost_his = [], [], []
        for frac in FRACS:
            pt, lo, hi = wp_auc_and_ci(ost_by_ds[ds], f"d{frac}")
            ost_aucs.append(pt); ost_los.append(lo); ost_his.append(hi)
        ost_aucs = np.array(ost_aucs); ost_los = np.array(ost_los); ost_his = np.array(ost_his)

        # Op-XGB
        op_aucs = op_los = op_his = None
        if opxgb_by_ds[ds]:
            depth_aucs, ci_lo, ci_hi = [], [], []
            for frac in FRACS:
                pt, lo, hi = wp_auc_and_ci(opxgb_by_ds[ds], f"d{frac}")
                depth_aucs.append(pt); ci_lo.append(lo); ci_hi.append(hi)
            op_aucs = np.array(depth_aucs); op_los = np.array(ci_lo); op_his = np.array(ci_hi)

        # LLM
        l_aucs = l_los = l_his = None
        if llm_by_ds[ds]:
            depth_aucs, ci_lo, ci_hi = [], [], []
            for frac in FRACS:
                pt, lo, hi = wp_auc_and_ci(llm_by_ds[ds], f"d{frac}")
                depth_aucs.append(pt); ci_lo.append(lo); ci_hi.append(hi)
            l_aucs = np.array(depth_aucs); l_los = np.array(ci_lo); l_his = np.array(ci_hi)

        # Plot CIs (background) before lines so lines sit on top
        ax.fill_between(x, ost_los, ost_his, alpha=0.15, color=OST_COLOR)
        if op_aucs is not None:
            ax.fill_between(x, op_los, op_his, alpha=0.10, color=OPXGB_COLOR)
        if l_aucs is not None:
            ax.fill_between(x, l_los, l_his, alpha=0.10, color=LLM_COLOR)

        ax.plot(x, ost_aucs, "o-", color=OST_COLOR, lw=2.0, ms=5.5, zorder=4,
                label="OST (operators)")
        if op_aucs is not None:
            ax.plot(x, op_aucs, "^--", color=OPXGB_COLOR, lw=1.5, ms=4.5, zorder=3,
                    label="Op-XGB-Early (ops + text)")
        if l_aucs is not None:
            ax.plot(x, l_aucs, "s--", color=LLM_COLOR, lw=1.4, ms=4.0, zorder=2,
                    label="SelfCheck (text)")

        # Smart final-value annotations: stack vertically when adjacent labels overlap
        end_pts = []
        end_pts.append((ost_aucs[-1], OST_COLOR))
        if op_aucs is not None:
            end_pts.append((op_aucs[-1], OPXGB_COLOR))
        if l_aucs is not None:
            end_pts.append((l_aucs[-1], LLM_COLOR))
        # Sort by value, then offset upward when a neighbor is too close
        end_pts.sort(key=lambda t: t[0])
        # Determine y range to compute "too close" threshold
        hi_max = max(np.nanmax(ost_his),
                     np.nanmax(op_his) if op_his is not None else 0,
                     np.nanmax(l_his)  if l_his is not None  else 0)
        y_max = min(1.0, hi_max + 0.07 * (hi_max - 0.48))
        y_min = 0.48
        gap = 0.022 * (y_max - y_min)
        adjusted = []
        prev_y = -1.0
        for val, col in end_pts:
            y = val
            if y - prev_y < gap:
                y = prev_y + gap
            adjusted.append((val, col, y))
            prev_y = y
        for val, col, y_text in adjusted:
            ax.annotate(f"{val:.3f}", xy=(100, val),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=7.5, color=col, va="center", ha="left",
                        annotation_clip=False)
            # tiny tick to point from data point to label if shifted
            if abs(y_text - val) > 1e-3:
                ax.plot([100, 104], [val, y_text], color=col, lw=0.6,
                        alpha=0.55, clip_on=False, zorder=5)
                # rewrite annotation at adjusted y
                ax.texts[-2].remove()  # remove the un-shifted one we just added
                ax.annotate(f"{val:.3f}", xy=(104, y_text),
                            xytext=(2, 0), textcoords="offset points",
                            fontsize=7.5, color=col, va="center", ha="left",
                            annotation_clip=False)

        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.55, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(5, 108)
        ax.set_xticks(FRACS)
        ax.set_xticklabels([f"{v}%" for v in FRACS], fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight="bold",
                     color=TITLE_COLORS[ds], pad=5)
        ax.set_xlabel("Trace fraction seen", fontsize=9)
        if ds == DATASETS[0]:
            ax.set_ylabel("WP-AUC", fontsize=9)
            ax.text(6, 0.484, "chance", fontsize=7, color="gray", va="bottom")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend on first axis
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_early_prediction.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
