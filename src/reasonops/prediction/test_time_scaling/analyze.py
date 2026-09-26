"""Evaluate selection and restart policies on saved attempts."""

import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from .config import HERE
from .ost_live import OSTScorer, segment

N_BOOT = 10000
POLICIES = [
    "single",
    "sequential",
    "vote",
    "len_select",
    "ost_select",
    "ost_restart",
    "ost_restart_stop",
    "oracle",
]
_BOX = re.compile("\\\\boxed\\{([^{}]+)\\}")


def aime_answer(text):
    m = _BOX.findall(text or "")
    if not m:
        return None
    d = re.sub("[^\\d]", "", m[-1])
    return d.lstrip("0") or "0" if d else None


def load(arm):
    runs = {}
    for f in sorted((HERE / "runs" / arm).rglob("*.json")):
        r = json.loads(f.read_text())
        runs[r["model"], r["dataset"], str(r["problem_id"])] = r
    return runs


def score_attempts(runs):
    scorer = OSTScorer(str(HERE / "ost_scorer_holdout.pt"))
    flat = [a for r in runs.values() for a in r["attempts"]]
    probs = scorer.score_batch([segment(a["reasoning"]) for a in flat])
    for a, p in zip(flat, probs):
        a["ost"] = p


def usable(attempts):
    return [a for a in attempts if not a["aborted"] and (not a.get("error"))]


def policies_for(key, base, ost):
    out = {}
    done = usable(base["attempts"])
    if done:
        out["single"] = float(done[0]["correct"])
        out["sequential"] = float(np.mean([a["correct"] for a in done]))
        out["len_select"] = float(max(done, key=lambda a: len(a["reasoning"]))["correct"])
        out["ost_select"] = float(max(done, key=lambda a: a["ost"])["correct"])
        out["oracle"] = float(any((a["correct"] for a in done)))
        if key[1] == "aime":
            ans = [(aime_answer(a["response"] or a["reasoning"]), a["correct"]) for a in done]
            ans = [(v, c) for v, c in ans if v is not None]
            if ans:
                cnt = Counter((v for v, _ in ans))
                top = max(cnt.values())
                tied = [v for v, c in cnt.items() if c == top]
                ok = {v for v, c in ans if c}
                out["vote"] = sum((v in ok for v in tied)) / len(tied)
            else:
                out["vote"] = 0.0
    if ost:
        surv = usable(ost["attempts"])
        if surv:
            out["ost_restart"] = float(max(surv, key=lambda a: a["ost"])["correct"])
            out["ost_restart_stop"] = float(surv[0]["correct"])
        else:
            out["ost_restart"] = out["ost_restart_stop"] = 0.0
    return out


def paired_ci(per_prob, pol, ref="single"):
    pairs = [(v[pol], v[ref]) for v in per_prob if pol in v and ref in v]
    if not pairs:
        return None
    d = np.array([a - b for a, b in pairs])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        len(d),
    )


def main():
    base, ost = (load("base"), load("ost"))
    print(f"arm base: {len(base)} runs, arm ost: {len(ost)} runs")
    if not base:
        sys.exit("no runs found")
    score_attempts(base)
    if ost:
        score_attempts(ost)
    rows = {k: policies_for(k, v, ost.get(k)) for k, v in base.items()}
    scopes = {
        "aime": lambda k: k[1] == "aime",
        "livecodebench": lambda k: k[1] == "livecodebench",
        "pooled": lambda k: True,
    }
    res = {}
    for scope, keep in scopes.items():
        by_prob = defaultdict(list)
        for k, v in rows.items():
            if keep(k):
                by_prob[k[1], k[2]].append(v)
        per_prob = [
            {
                p: float(np.mean([d[p] for d in ds if p in d]))
                for p in POLICIES
                if any((p in d for d in ds))
            }
            for ds in by_prob.values()
        ]
        res[scope] = {}
        for p in POLICIES:
            vals = [v[p] for v in per_prob if p in v]
            if not vals:
                continue
            res[scope][p] = {"acc": float(np.mean(vals)), "n": len(vals)}
            for ref, tag in (("single", "delta_vs_single"), ("sequential", "delta_vs_sequential")):
                ci = paired_ci(per_prob, p, ref)
                if ci and p != ref:
                    res[scope][p][tag] = {"mean": ci[0], "lo": ci[1], "hi": ci[2], "n": ci[3]}
    from sklearn.metrics import roc_auc_score

    def wp_auc(keyfn):
        aucs = []
        for r in base.values():
            done = usable(r["attempts"])
            ys = [a["correct"] for a in done]
            if len(set(ys)) == 2:
                aucs.append(roc_auc_score(ys, [keyfn(a) for a in done]))
        return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))

    a_ost, n = wp_auc(lambda a: a["ost"])
    a_len, _ = wp_auc(lambda a: len(a["reasoning"]))
    res["verifier"] = {"ost_wp_auc": a_ost, "length_wp_auc": a_len, "n_groups": n}
    print(f"verifier: OST WP-AUC {a_ost:.4f}, length control {a_len:.4f} (n={n})")

    def arm_stats(runs, label):
        if not runs:
            return {}
        att = [len(usable(r["attempts"])) for r in runs.values()]
        ab = [sum((a["aborted"] for a in runs[k]["attempts"])) for k in runs]
        tok = [r["tokens_used"] for r in runs.values()]
        s = {
            "mean_tokens": float(np.mean(tok)),
            "mean_completed_attempts": float(np.mean(att)),
            "mean_aborts": float(np.mean(ab)),
            "n_runs": len(runs),
        }
        print(f"{label}: {s}")
        return s

    res["_arm_base"] = arm_stats(base, "base")
    res["_arm_ost"] = arm_stats(ost, "ost")
    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
