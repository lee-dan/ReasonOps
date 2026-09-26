"""Calibrate a prefix-based abort rule on non-evaluation problems."""

import gzip
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from reasonops.paths import DATA_DIR

from .config import HERE
from .ost_live import OSTScorer, segment
from .train_scorer import pool_problem_ids

CHECK = 2000
GRID = {"min": [2000, 4000], "consec": [2, 3], "tau": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
N_PER_DS = 2000
DATASETS = {"aime", "livecodebench"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", default="ost_scorer_holdout.pt")
    scorer_path = parser.parse_args().scorer
    held_out = pool_problem_ids()
    rng = np.random.default_rng(42)
    pool = defaultdict(list)
    with gzip.open(DATA_DIR / "final_dataset.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if (
                r.get("dataset") in DATASETS
                and r.get("correct") is not None
                and (str(r.get("problem_id")) not in held_out)
                and (len(r.get("reasoning") or "") >= CHECK)
            ):
                pool[r["dataset"]].append((r["reasoning"], int(bool(r["correct"]))))
    sample = []
    for ds, items in pool.items():
        idx = rng.choice(len(items), size=min(N_PER_DS, len(items)), replace=False)
        sample += [items[i] for i in idx]
    print(f"{len(sample)} corpus traces ({ {ds: len(v) for ds, v in pool.items()} })", flush=True)
    marks_per, seqs = ([], [])
    for reasoning, _ in sample:
        marks = list(range(CHECK, len(reasoning) + 1, CHECK))
        marks_per.append(marks)
        seqs += [segment(reasoning[:m]) for m in marks]
    print(f"scoring {len(seqs)} prefixes", flush=True)
    probs = OSTScorer(str(HERE / scorer_path)).score_batch(seqs)
    traces, i = ([], 0)
    for (reasoning, correct), marks in zip(sample, marks_per):
        traces.append(
            {
                "correct": correct,
                "chars": len(reasoning),
                "marks": marks,
                "scores": probs[i : i + len(marks)],
            }
        )
        i += len(marks)
    auc = {}
    for M in [2000, 4000, 8000, 16000]:
        ys, ss = zip(
            *[
                (t["correct"], t["scores"][min(M // CHECK, len(t["scores"])) - 1])
                for t in traces
                if t["chars"] >= M
            ]
        )
        auc[M] = round(roc_auc_score(ys, ss), 4)
    print("prefix AUC by milestone:", auc, flush=True)

    def fire_at(t, mn, k, tau):
        below = 0
        for m, s in zip(t["marks"], t["scores"]):
            if m < mn:
                continue
            below = below + 1 if s < tau else 0
            if below >= k:
                return m
        return None

    grid = []
    for mn in GRID["min"]:
        for k in GRID["consec"]:
            for tau in GRID["tau"]:
                fires = [(t, fire_at(t, mn, k, tau)) for t in traces]
                corr = [f for t, f in fires if t["correct"]]
                inc = [(t, f) for t, f in fires if not t["correct"]]
                fa = float(np.mean([f is not None for f in corr]))
                catch = float(np.mean([f is not None for _, f in inc]))
                save = float(np.mean([1 - f / t["chars"] for t, f in inc if f is not None]) or 0)
                grid.append(
                    {
                        "min": mn,
                        "consec": k,
                        "tau": tau,
                        "false_abort": round(fa, 3),
                        "catch": round(catch, 3),
                        "save": round(save, 3),
                    }
                )
    ok = [g for g in grid if g["false_abort"] <= 0.1]
    chosen = max(ok, key=lambda g: g["catch"]) if ok else min(grid, key=lambda g: g["false_abort"])
    print("chosen rule:", chosen)
    (HERE / "calibration.json").write_text(
        json.dumps(
            {"check_chars": CHECK, "prefix_auc": auc, "grid": grid, "chosen": chosen}, indent=2
        )
    )
    print(f"wrote {HERE / 'calibration.json'}")


if __name__ == "__main__":
    main()
