"""Fit operator-feature scorers with problem-held-out cross-validation."""

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from reasonops.prediction.op_seq_baseline import op_features

from .config import HERE
from .ost_live import OSTScorer, segment

N_FOLDS = 8
N_BOOT = 10000
SEED = 0


def load_attempts():
    rows = []
    for f in sorted((HERE / "runs/base").rglob("*.json")):
        r = json.loads(f.read_text())
        for a in r["attempts"]:
            if a["aborted"] or a.get("error"):
                continue
            seq = segment(a["reasoning"])
            rows.append(
                {
                    "dataset": r["dataset"],
                    "model": r["model"],
                    "problem_id": str(r["problem_id"]),
                    "idx": a["idx"],
                    "correct": int(a["correct"]),
                    "capped": bool(a.get("capped")),
                    "tokens": a["tokens"],
                    "chars": len(a["reasoning"]),
                    "seq": seq,
                    "reasoning": a["reasoning"],
                }
            )
    return rows


def fit_oof(rows, featfn, seed=SEED):
    """Fit each fold without any attempts from its evaluation problems."""
    from reasonops.utils import make_clf

    X = np.array([featfn(r) for r in rows], dtype=float)
    y = np.array([r["correct"] for r in rows])
    g = np.array([r["problem_id"] for r in rows])
    oof = np.zeros(len(rows))
    n = min(N_FOLDS, len(set(g)))
    if n < 2:
        raise ValueError("Scoring requires at least two distinct problems")
    for tr, te in GroupKFold(n_splits=n).split(X, y, g):
        if len(set(y[tr])) < 2:
            oof[te] = 0.5
            continue
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def within_cell(rows, scores):
    cells = defaultdict(list)
    for r, s in zip(rows, scores):
        cells[r["model"], r["problem_id"]].append((r["correct"], s))
    out = {}
    for k, v in cells.items():
        if len({c for c, _ in v}) == 2:
            out[k] = roc_auc_score([c for c, _ in v], [s for _, s in v])
    return out


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    if not len(d):
        return None
    rng = np.random.default_rng(SEED)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_cells": len(d),
    }


def analyse(rows, tag, ost_scores, res):
    rng = np.random.default_rng(SEED)
    shuffled = []
    for r in rows:
        s = list(r["seq"])
        rng.shuffle(s)
        shuffled.append(s)
    scorers = {
        "op": fit_oof(rows, lambda r: op_features(r["seq"])),
        "length": np.array([r["chars"] for r in rows], dtype=float),
        "n_ops": np.array([len(r["seq"]) for r in rows], dtype=float),
        "shuffled": fit_oof(
            [{**r, "seq": s} for r, s in zip(rows, shuffled)], lambda r: op_features(r["seq"])
        ),
        "ost": ost_scores,
    }
    cellauc = {k: within_cell(rows, v) for k, v in scorers.items()}
    n_cells = len(cellauc["op"])
    res[tag] = {
        "n_attempts": len(rows),
        "n_mixed_cells": n_cells,
        "auc": {
            k: float(np.mean(list(v.values()))) if v else float("nan") for k, v in cellauc.items()
        },
    }
    res[tag]["paired_vs_length"] = {
        k: paired(cellauc[k], cellauc["length"]) for k in ("op", "n_ops", "shuffled", "ost")
    }
    res[tag]["op_vs_shuffled"] = paired(cellauc["op"], cellauc["shuffled"])
    res[tag]["op_vs_n_ops"] = paired(cellauc["op"], cellauc["n_ops"])
    return (cellauc, scorers)


def main():
    rows = load_attempts()
    print(f"{len(rows)} usable attempts", flush=True)
    ost = OSTScorer(str(HERE / "ost_scorer_holdout.pt"))
    all_ost = np.array(ost.score_batch([r["seq"] for r in rows]))
    res = {}
    keep = {}
    for ds in ("aime", "livecodebench"):
        sub = [(r, s) for r, s in zip(rows, all_ost) if r["dataset"] == ds]
        full = [r for r, _ in sub]
        prim = [(r, s) for r, s in sub if not r["capped"]]
        c, sc = analyse(
            [r for r, _ in prim], f"{ds} (capped excluded)", np.array([s for _, s in prim]), res
        )
        keep[ds] = ([r for r, _ in prim], sc)
        analyse(full, f"{ds} (all attempts, sensitivity)", np.array([s for _, s in sub]), res)
    (HERE / "scorer_metrics.json").write_text(json.dumps(res, indent=2))
    dump = {
        ds: [
            {
                "model": r["model"],
                "problem_id": r["problem_id"],
                "idx": r["idx"],
                "correct": r["correct"],
                "tokens": r["tokens"],
                "chars": r["chars"],
                "op": float(s),
            }
            for r, s in zip(rw, sc["op"])
        ]
        for ds, (rw, sc) in keep.items()
    }
    (HERE / "pool_scores.json").write_text(json.dumps(dump))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
