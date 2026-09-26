"""Estimate best-of-N accuracy and paired confidence intervals."""

import json
from collections import defaultdict

import numpy as np

from .config import HERE

NS = [1, 2, 3, 5, 8, 12]
N_SIM = 2000
N_BOOT = 5000
SEED = 0


def load_cells():
    d = json.loads((HERE / "pool_scores.json").read_text())
    out = defaultdict(list)
    for ds, rows in d.items():
        for r in rows:
            out[ds, r["model"], r["problem_id"]].append(r)
    return out


def bon_per_cell(cells, keyfn, rng, N, random_key=False):
    """Resample N candidates with replacement within each model–problem cell."""
    out = {}
    for key, atts in cells.items():
        if len(atts) < 2:
            continue
        c = np.array([a["correct"] for a in atts], dtype=float)
        idx = rng.integers(0, len(atts), (N_SIM, N))
        k = (
            rng.random((N_SIM, N))
            if random_key
            else np.array([keyfn(a) for a in atts], dtype=float)[idx]
        )
        pick = np.take_along_axis(idx, np.argmax(k, axis=1)[:, None], axis=1)[:, 0]
        out[key] = float(c[pick].mean())
    return out


def paired_over_problems(a, b, seed=SEED):
    """Average model differences within each problem, then bootstrap problems."""
    per = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        per[key[2]].append(a[key] - b[key])
    d = np.array([np.mean(v) for v in per.values()])
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": len(d),
    }


def bon_curve(cells, keyfn, rng, random_key=False):
    curve = {}
    for N in NS:
        vals = []
        for atts in cells.values():
            if len(atts) < 2:
                continue
            c = np.array([a["correct"] for a in atts], dtype=float)
            idx = rng.integers(0, len(atts), (N_SIM, N))
            k = (
                rng.random((N_SIM, N))
                if random_key
                else np.array([keyfn(a) for a in atts], dtype=float)[idx]
            )
            pick = np.take_along_axis(idx, np.argmax(k, axis=1)[:, None], axis=1)[:, 0]
            vals.append(float(c[pick].mean()))
        curve[N] = float(np.mean(vals))
    return curve


def main():
    cells = load_cells()
    rng = np.random.default_rng(SEED)
    res = {}
    for ds in ("aime", "livecodebench"):
        sub = {k: v for k, v in cells.items() if k[0] == ds}
        if not sub:
            continue
        pi = float(
            np.mean([np.mean([a["correct"] for a in v]) for v in sub.values() if len(v) >= 2])
        )
        scorers = {
            "operator": lambda a: a["op"],
            "length": lambda a: a["chars"],
            "random": lambda a: rng.random(),
        }
        res[ds] = {"base_rate": pi, "curves": {}}
        for name, fn in scorers.items():
            cur = bon_curve(sub, fn, rng, random_key=name == "random")
            res[ds]["curves"][name] = cur
        res[ds]["paired"] = {}
        for N in NS[1:]:
            po = bon_per_cell(sub, lambda a: a["op"], np.random.default_rng(SEED), N)
            pl = bon_per_cell(sub, lambda a: a["chars"], np.random.default_rng(SEED), N)
            pr = bon_per_cell(sub, None, np.random.default_rng(SEED), N, random_key=True)
            dr, dl = (paired_over_problems(po, pr), paired_over_problems(po, pl))
            res[ds]["paired"][N] = {"vs_random": dr, "vs_length": dl}
        lo_n, hi_n = (NS[2], NS[-1])
        did = {}
        for nm, fn, rk in (
            ("operator", lambda a: a["op"], False),
            ("length", lambda a: a["chars"], False),
            ("random", None, True),
        ):
            a_hi = bon_per_cell(sub, fn, np.random.default_rng(SEED), hi_n, random_key=rk)
            a_lo = bon_per_cell(sub, fn, np.random.default_rng(SEED), lo_n, random_key=rk)
            did[nm] = paired_over_problems(a_hi, a_lo)
        res[ds]["scaling"] = did
        for nm in ("operator", "length", "random"):
            did[nm]
    (HERE / "best_of_n.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
