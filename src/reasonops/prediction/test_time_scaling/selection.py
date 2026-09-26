"""Evaluate operator-ranked selection and subset voting."""

import json
from collections import defaultdict

import numpy as np

from .analyze import aime_answer
from .config import HERE

N_BOOT = 10000
KS = [3, 5, 10]
N_RANDOM = 200


def load_joined():
    scores = json.loads((HERE / "pool_scores.json").read_text())
    text = {}
    for f in sorted((HERE / "runs/base").rglob("*.json")):
        r = json.loads(f.read_text())
        for a in r["attempts"]:
            text[r["dataset"], r["model"], str(r["problem_id"]), a["idx"]] = a
    cells = defaultdict(list)
    for ds, rows in scores.items():
        for r in rows:
            a = text.get((ds, r["model"], r["problem_id"], r["idx"]))
            if a is None:
                continue
            cells[ds, r["model"], r["problem_id"]].append(
                {
                    **r,
                    "answer": aime_answer(a["response"] or a["reasoning"])
                    if ds == "aime"
                    else None,
                }
            )
    return cells


def tally(atts, weights=None):
    a = [
        (x["answer"], x["correct"], 1.0 if weights is None else weights[i])
        for i, x in enumerate(atts)
        if x["answer"] is not None
    ]
    if not a:
        return 0.0
    w = defaultdict(float)
    for v, _, wt in a:
        w[v] += wt
    top = max(w.values())
    tied = [v for v, x in w.items() if x == top]
    ok = {v for v, c, _ in a if c}
    return sum((v in ok for v in tied)) / len(tied)


def policies(atts, rng, voting):
    o = {
        "random": float(np.mean([a["correct"] for a in atts])),
        "length": float(max(atts, key=lambda a: a["chars"])["correct"]),
        "op": float(max(atts, key=lambda a: a["op"])["correct"]),
        "oracle": float(any((a["correct"] for a in atts))),
    }
    if voting:
        o["vote"] = tally(atts)
        o["vote_op_weighted"] = tally(atts, [a["op"] for a in atts])
        by = sorted(atts, key=lambda a: -a["op"])
        for k in KS:
            kk = min(k, len(atts))
            o[f"vote_top{k}"] = tally(by[:kk])
            o[f"vote_rand{k}"] = float(
                np.mean(
                    [
                        tally([atts[i] for i in rng.choice(len(atts), kk, replace=False)])
                        for _ in range(N_RANDOM)
                    ]
                )
            )
    return o


def paired(per_prob, a, b, seed=0):
    d = np.array([v[a] - v[b] for v in per_prob if a in v and b in v])
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


def main():
    cells = load_joined()
    if not cells:
        raise ValueError("No scored attempts found")
    rng = np.random.default_rng(0)
    res = {}
    for ds in ("aime", "livecodebench"):
        sub = {k: v for k, v in cells.items() if k[0] == ds}
        if not sub:
            continue
        by_prob = defaultdict(list)
        for k, atts in sub.items():
            by_prob[k[2]].append(policies(atts, rng, voting=ds == "aime"))
        per_prob = [
            {p: float(np.mean([d[p] for d in ds_ if p in d])) for p in {p for d in ds_ for p in d}}
            for ds_ in by_prob.values()
        ]
        names = (
            ["random", "length", "op", "vote", "vote_op_weighted"]
            + [f"vote_top{k}" for k in KS]
            + [f"vote_rand{k}" for k in KS]
            + ["oracle"]
        )
        res[ds] = {}
        for n in names:
            vals = [v[n] for v in per_prob if n in v]
            if not vals:
                continue
            d = paired(per_prob, n, "random")
            res[ds][n] = {"acc": float(np.mean(vals)), "delta_vs_random": d, "n": len(vals)}
            "—" if not d or n == "random" else f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
        for a, b, why in [
            ("op", "length", "operator scorer vs the length control"),
            ("vote_op_weighted", "vote", "weighting the vote by operators"),
            ("op", "vote", "selection vs plain self-consistency"),
        ]:
            d = paired(per_prob, a, b)
            if d:
                res[ds][f"{a}_vs_{b}"] = d
        for k in KS:
            d = paired(per_prob, f"vote_top{k}", f"vote_rand{k}")
            if d:
                res[ds][f"top{k}_vs_rand{k}"] = d
    for ds, d in res.items():
        orc = d["oracle"]["acc"]
        for n, v in d.items():
            if not isinstance(v, dict) or "acc" not in v:
                continue
            assert v["n"] == d["oracle"]["n"], (
                f"{ds}/{n} covers {v['n']} problems, oracle {d['oracle']['n']}"
            )
            assert v["acc"] <= orc + 1e-09, f"{ds}/{n} acc {v['acc']:.3f} exceeds oracle {orc:.3f}"
    print("invariants OK: equal denominators, nothing above the oracle")
    (HERE / "selection.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
