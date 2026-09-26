#!/usr/bin/env python3
"""
Compute per-dataset and global WP-AUC from all local prediction files.
Outputs data/wpauc_summary.json with every number the paper cites.

Global WP-AUC uses pooled computation (all problems across datasets treated as
one pool, per-problem AUC averaged) — this matches the paper's reported global values.

Usage:
    python -m reasonops.eval.compute_summaries
"""
import gzip
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score

from reasonops.utils import EXCLUDE_DATASETS, FRACS

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

DATASETS_6 = ["aime", "arc_challenge", "gpqa", "livecodebench", "math", "mmlu_pro"]
_FRAC_PCTS = [int(f * 100) for f in FRACS]  # (10, 25, 50, 75, 100) for column key names


def load_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def wp_auc_by_dataset(rows, score_key, datasets):
    """Per-dataset WP-AUC + pooled global (paper convention)."""
    # Per dataset
    by_ds = defaultdict(list)
    for r in rows:
        v = r.get(score_key)
        ds = r.get("dataset", "")
        if v is not None and ds in datasets:
            by_ds[ds].append((r["problem_id"], r["correct"], v))

    result = {}
    for ds, entries in by_ds.items():
        groups = defaultdict(list)
        for pid, y, p in entries:
            groups[pid].append((y, p))
        aucs = [
            roc_auc_score([x[0] for x in v], [x[1] for x in v])
            for v in groups.values()
            if len(set(x[0] for x in v)) == 2
        ]
        result[ds] = round(float(np.mean(aucs)), 4) if aucs else None

    # Pooled global (all problems across all datasets, equal per-problem weight)
    all_groups = defaultdict(list)
    for r in rows:
        v = r.get(score_key)
        ds = r.get("dataset", "")
        if v is not None and ds in datasets:
            key = ds + "::" + r["problem_id"]
            all_groups[key].append((r["correct"], v))
    all_aucs = [
        roc_auc_score([x[0] for x in v], [x[1] for x in v])
        for v in all_groups.values()
        if len(set(x[0] for x in v)) == 2
    ]
    result["global"] = round(float(np.mean(all_aucs)), 4) if all_aucs else None
    result["n_problems_global"] = len(all_aucs)
    return result


def compute_model_id_summary():
    path = DATA / "model_id" / "model_id_predictions.jsonl.gz"
    if not path.exists():
        return {"error": "not found"}

    rows = load_jsonl(path)
    if not rows:
        return {"error": "empty file"}

    sample = rows[0]
    # Keys: trace_id, dataset, model, problem_id, model_id_pred, model_id_prob, model_id_probs
    ys_true = [r.get("model") for r in rows]
    ys_pred = [r.get("model_id_pred") for r in rows]

    valid = [(t, p) for t, p in zip(ys_true, ys_pred) if t is not None and p is not None]
    acc = None
    if valid:
        acc = round(sum(1 for t, p in valid if t == p) / len(valid), 4)

    # Macro-AUC from model_id_probs
    macro_auc = None
    try:
        labels = sorted(set(r["model"] for r in rows if r.get("model")))
        y_true_enc = [labels.index(r["model"]) for r in rows if r.get("model") and r.get("model_id_probs")]
        proba_rows = [r for r in rows if r.get("model") and r.get("model_id_probs")]
        proba_mat = np.array([[r["model_id_probs"].get(l, 0.0) for l in labels] for r in proba_rows])
        from sklearn.metrics import roc_auc_score as ras
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true_enc, classes=list(range(len(labels))))
        macro_auc = round(float(ras(y_bin, proba_mat, average="macro", multi_class="ovr")), 4)
    except Exception as e:
        macro_auc = f"error: {e}"

    return {
        "source": "data/model_id/model_id_predictions.jsonl.gz",
        "n": len(rows),
        "n_valid": len(valid),
        "accuracy": acc,
        "macro_auc": macro_auc,
        "n_classes": len(set(ys_true)) if ys_true else None,
    }


def compute_stability_summary():
    path = DATA / "stability" / "stability_cache.npz"
    if not path.exists():
        return {"error": "stability_cache.npz not found"}

    cache = np.load(path, allow_pickle=True)
    real_embs = cache["real_embs"]   # (K, seeds, dim)
    rand_embs = cache["rand_embs"]   # (K_fake, seeds, dim)
    K, S, dim = real_embs.shape

    within = []
    for c in range(K):
        mat = real_embs[c]
        sim = mat @ mat.T
        idx = np.triu_indices(S, k=1)
        within.extend(sim[idx].tolist())

    rng = np.random.default_rng(7)
    flat = real_embs.reshape(-1, dim)
    labels = np.repeat(np.arange(K), S)
    across = []
    while len(across) < 2000:
        i, j = rng.integers(0, len(flat), size=2)
        if labels[i] != labels[j]:
            across.append(float(flat[i] @ flat[j]))

    rand_within = []
    for c in range(rand_embs.shape[0]):
        mat = rand_embs[c]
        sim = mat @ mat.T
        idx = np.triu_indices(S, k=1)
        rand_within.extend(sim[idx].tolist())

    w  = np.array(within)
    ac = np.array(across[:2000])
    rw = np.array(rand_within)

    from scipy.stats import mannwhitneyu
    def rank_biserial(a, b):
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        r = 2 * stat / (len(a) * len(b)) - 1
        return float(r), float(p)

    r_wa, p_wa = rank_biserial(w, ac)
    r_wr, p_wr = rank_biserial(w, rw)

    return {
        "source": "data/stability/stability_cache.npz",
        "K": int(K), "seeds": int(S),
        "within_n": len(w), "across_n": len(ac), "random_n": len(rw),
        "within_median": round(float(np.median(w)), 3),
        "across_median": round(float(np.median(ac)), 3),
        "random_median": round(float(np.median(rw)), 3),
        "r_rb_within_vs_across": round(r_wa, 3),
        "p_within_vs_across": float(p_wa),
        "r_rb_within_vs_random": round(r_wr, 3),
        "p_within_vs_random": float(p_wr),
    }


def compute_kappa_summary():
    out = {}
    for path in sorted((DATA / "kappa").glob("kappa_*.json")):
        with open(path) as f:
            d = json.load(f)
        name = path.stem
        if isinstance(d, list):
            correct = sum(1 for r in d if r.get("predicted") == r.get("label"))
            out[name] = {
                "accuracy": round(correct / len(d), 4) if d else None,
                "n": len(d),
                "source": f"data/kappa/{path.name}",
            }
        elif isinstance(d, dict) and "accuracy" in d:
            out[name] = {"accuracy": round(d["accuracy"], 4), "source": f"data/kappa/{path.name}"}
        else:
            out[name] = {"raw_keys": list(d.keys()) if isinstance(d, dict) else str(type(d)),
                         "source": f"data/kappa/{path.name}"}
    return out


def main():
    out = {}

    # --- Kappa / judge accuracy ---
    print("Computing kappa summary...")
    out["kappa"] = compute_kappa_summary()
    for name, v in out["kappa"].items():
        print(f"  {name}: {v.get('accuracy')} (n={v.get('n')})")

    # --- OST WP-AUC ---
    print("\nComputing OST WP-AUC...")
    ost_rows   = load_jsonl(DATA / "ost_cv" / "preds_ost.jsonl.gz")
    ost_rows_w = load_jsonl(DATA / "ost_cv" / "preds_ost_within.jsonl.gz")
    ost_cd, ost_id = {}, {}
    for frac in _FRAC_PCTS:
        ost_cd[f"d{frac}"] = wp_auc_by_dataset(ost_rows,   f"d{frac}", DATASETS_6)
        ost_id[f"d{frac}"] = wp_auc_by_dataset(ost_rows_w, f"d{frac}", DATASETS_6)
    out["ost"] = {
        "source_cross":  "data/ost_cv/preds_ost.jsonl.gz",
        "source_within": "data/ost_cv/preds_ost_within.jsonl.gz",
        "cross_dataset":   ost_cd,
        "in_distribution": ost_id,
    }

    # --- Op-XGB + scalar baselines from merged all_predictions.jsonl.gz ---
    # (op_seq_tfidf_* columns match paper numbers; scalar columns also from here)
    print("\nComputing Op-XGB and scalar baselines from all_predictions.jsonl.gz...")
    all_pred_rows = load_jsonl(DATA / "predictions" / "all_predictions.jsonl.gz")
    print(f"  {len(all_pred_rows):,} rows")

    out["op_xgb"] = {
        "source": "data/predictions/all_predictions.jsonl.gz",
        "note": "columns: op_seq_tfidf_cross, op_seq_tfidf_within",
        "cross_dataset":   wp_auc_by_dataset(all_pred_rows, "op_seq_tfidf_cross",  DATASETS_6),
        "in_distribution": wp_auc_by_dataset(all_pred_rows, "op_seq_tfidf_within", DATASETS_6),
    }

    for name, col_id, col_cd in [
        ("length",    "length_within",    "length_cross"),
        ("backtrack", "backtrack_within", "backtrack_cross"),
        ("wait",      "wait_within",      "wait_cross"),
    ]:
        out[name] = {
            "source": "data/predictions/all_predictions.jsonl.gz",
            "in_distribution": wp_auc_by_dataset(all_pred_rows, col_id, DATASETS_6),
            "cross_dataset":   wp_auc_by_dataset(all_pred_rows, col_cd, DATASETS_6),
        }

    # --- LLM-judge SelfCheck (full 6-dataset coverage) ---
    # Source: preds_llm_judge_priority.jsonl (AIME, GPQA, HumanEval)
    #         preds_llm_judge_remaining.jsonl (LCB, ARC, MATH, MMLU-Pro)
    # Score column: llm_judge_prob
    print("\nComputing LLM-judge SelfCheck WP-AUC...")
    ljp_files = [
        DATA / "predictions" / "preds_llm_judge_priority.jsonl",
        DATA / "predictions" / "preds_llm_judge_remaining.jsonl",
    ]
    ljp_rows = []
    for pf in ljp_files:
        for r in load_jsonl(pf):
            if r.get("llm_judge_prob") is not None:
                ljp_rows.append(r)
    print(f"  {len(ljp_rows):,} rows with llm_judge_prob")
    ljp_cd = wp_auc_by_dataset(ljp_rows, "llm_judge_prob", DATASETS_6)

    out["llm_partial"] = {
        "source_files": [f"data/predictions/{pf.name}" for pf in ljp_files],
        "n_rows": len(ljp_rows),
        "cross_dataset": {"d100": ljp_cd},
    }

    # --- Stability ---
    print("\nComputing stability stats...")
    out["stability"] = compute_stability_summary()

    # --- Model ID ---
    print("\nComputing model ID stats...")
    out["model_id"] = compute_model_id_summary()

    # --- Save ---
    out_path = DATA / "wpauc_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # --- Print paper table ---
    print("\n" + "="*80)
    print("PAPER TABLE: Correctness Prediction WP-AUC")
    print("Format: CD / ID  (cross-dataset / in-distribution)")
    print("="*80)

    cols = [
        ("Length",    out["length"],    "cross_dataset", "in_distribution"),
        ("Backtrack", out["backtrack"], "cross_dataset", "in_distribution"),
        ("Wait",      out["wait"],      "cross_dataset", "in_distribution"),
        ("Op-XGB",    out["op_xgb"],    "cross_dataset", "in_distribution"),
        ("OST",       out["ost"],       "cross_dataset.d100", "in_distribution.d100"),
    ]

    def get_nested(d, key):
        parts = key.split(".")
        v = d
        for p in parts:
            v = v[p]
        return v

    header = f"{'Dataset':<18}"
    for label, *_ in cols:
        header += f"  {label:>14}"
    print(header)
    print("-"*80)

    for ds in DATASETS_6 + ["global"]:
        row = f"{ds:<18}"
        for label, d, cd_key, id_key in cols:
            cd = get_nested(d, cd_key).get(ds)
            id_ = get_nested(d, id_key).get(ds)
            cd_s = f"{cd:.3f}" if cd is not None else "  N/A"
            id_s = f"{id_:.3f}" if id_ is not None else "  N/A"
            row += f"  {cd_s}/{id_s}"
        print(row)

    print("\nOST early prediction (CD, global):")
    for frac in _FRAC_PCTS:
        v = out["ost"]["cross_dataset"][f"d{frac}"]["global"]
        print(f"  d{frac:3d}%: {v:.4f}")

    print("\nLLM-judge SelfCheck (CD, global, d100):")
    v = out["llm_partial"]["cross_dataset"]["d100"]["global"]
    print(f"  d100%: {v:.4f}")

    stab = out["stability"]
    print(f"\nStability:")
    print(f"  within median:    {stab.get('within_median')}")
    print(f"  across median:    {stab.get('across_median')}")
    print(f"  random median:    {stab.get('random_median')}")
    print(f"  r_rb (w vs ac):  {stab.get('r_rb_within_vs_across')}")
    print(f"  r_rb (w vs rnd): {stab.get('r_rb_within_vs_random')}")
    print(f"  p (w vs ac):     {stab.get('p_within_vs_across'):.2e}")
    print(f"  p (w vs rnd):    {stab.get('p_within_vs_random'):.2e}")

    mid = out["model_id"]
    print(f"\nModel ID:")
    print(f"  accuracy:   {mid.get('accuracy')}")
    print(f"  macro-AUC:  {mid.get('macro_auc')}")

    print(f"\nKappa:")
    for name, v in out["kappa"].items():
        print(f"  {name}: acc={v.get('accuracy')}")


if __name__ == "__main__":
    main()
