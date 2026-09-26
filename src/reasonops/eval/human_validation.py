"""Compute pipeline agreement and inter-annotator agreement from a label CSV."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

OPERATORS = [
    "INITIATING",
    "QUALIFYING",
    "GROUNDING",
    "INFERRING",
    "HYPOTHESIZING",
    "BACKTRACKING",
    "CONSTRAINING",
]
LABELS = OPERATORS + ["NONE"]


def normalize(raw):
    raw = raw.strip().upper()
    if not raw:
        return ""
    matches = [op for op in LABELS if op.startswith(raw)]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous operator: {raw!r}")
    return matches[0]


def compare(rows, reference, annotation, seed=20260924):
    pairs = [(r[reference], r[annotation]) for r in rows if r[reference] and r[annotation]]
    if not pairs:
        return {"n": 0}
    x, y = map(np.asarray, zip(*pairs))
    n = len(x)
    hit = int(sum(x == y))
    p, z = hit / n, 1.96
    d = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    cm = confusion_matrix(x, y, labels=LABELS)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, (cm / n).ravel(), size=20000).reshape(-1, 8, 8)
    po = np.trace(draws, axis1=1, axis2=2) / n
    pe = (draws.sum(1) * draws.sum(2)).sum(1) / n**2
    valid = pe < 1
    bootstrap = (po[valid] - pe[valid]) / (1 - pe[valid])
    pr, rec, f1, support = precision_recall_fscore_support(x, y, labels=OPERATORS, zero_division=0)
    kappa = float(cohen_kappa_score(x, y)) if len(set(x) | set(y)) > 1 else None
    return {
        "n": n,
        "matches": hit,
        "agreement": p,
        "wilson95": [(center - half) / d, (center + half) / d],
        "kappa": kappa,
        "kappa_bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist()
        if len(bootstrap)
        else None,
        "confusion_labels": LABELS,
        "confusion": cm.tolist(),
        "per_operator": {
            op: dict(n=int(support[i]), precision=pr[i], recall=rec[i], f1=f1[i])
            for i, op in enumerate(OPERATORS)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--annotators", nargs="+", default=["annotator_1", "annotator_2"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.labels.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate span IDs")
    columns = ["pipeline_label", *args.annotators]
    for row in rows:
        for column in columns:
            row[column] = normalize(row[column])
    reports = {f"{a}_vs_{b}": compare(rows, a, b) for a, b in combinations(columns, 2)}
    shared = [r for r in rows if all(r[c] for c in columns)]
    reports["shared"] = {
        f"pipeline_vs_{a}": compare(shared, "pipeline_label", a) for a in args.annotators
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
