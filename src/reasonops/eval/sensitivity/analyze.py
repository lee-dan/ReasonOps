"""Summarize sensitivity results and plot parameter sweeps."""

from reasonops.paths import DATA_DIR, OUTPUT_DIR

HERE = OUTPUT_DIR / "sensitivity"
import json
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RES = HERE / "results"
C_MID = "#0072B2"
C_WP = "#009E73"
C_KAP = "#D55E00"


def load_rows():
    rows = {}
    for f in sorted(RES.glob("sweep*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("quick") or r.get("error"):
                continue
            rows[r["axis"], r["tag"], r["seed"]] = r
    return list(rows.values())


def mstd(vals):
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(vals, prec=4):
    m, s = mstd(vals)
    return f"{m:.{prec}f} ± {s:.{prec}f}" if len(vals) > 1 else f"{m:.{prec}f}"


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return dict(sorted(g.items()))


def main():
    rows = load_rows()
    if not rows:
        raise ValueError("No full sweep results found")
    by_axis = group(rows, lambda r: r["axis"])
    kappa = {}
    for k in range(4, 13):
        f = DATA_DIR / f"k_sweep/kappa_k{k}.json"
        if f.exists():
            kappa[k] = json.loads(f.read_text())["kappa"]
    byK = group(by_axis.get("k", []), lambda r: r["K_req"])
    default_rows = by_axis.get("default", [])
    plen = group(by_axis.get("length", []) + default_rows, lambda r: r["n"])
    pfreq = group(by_axis.get("freq", []) + default_rows, lambda r: r["freq"])
    joint = by_axis.get("joint", [])
    (RES / "summary.json").write_text(json.dumps(by_axis, indent=2))
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))

    def line(a, xs, rows_by_x, key, color, marker, label, ls="-"):
        ms = [mstd([r[key] for r in rows_by_x[x]]) for x in xs]
        a.errorbar(
            xs,
            [m for m, _ in ms],
            yerr=[s for _, s in ms],
            fmt=marker,
            ls=ls,
            color=color,
            label=label,
            capsize=2.5,
            ms=4.5,
            lw=1.6,
        )

    a = ax[0, 0]
    Ks = list(byK)
    line(a, Ks, byK, "modelid_auc", C_MID, "o", "model-ID AUC")
    line(a, Ks, byK, "correct_wpauc", C_WP, "s", "correctness WP-AUC")
    kk = [k for k in Ks if k in kappa]
    a.plot(kk, [kappa[k] for k in kk], "^--", color=C_KAP, ms=4.5, lw=1.4, label="judge κ")
    a.axvline(7, color="grey", ls=":", lw=1)
    a.set_xlabel("K (number of operators)")
    a.set_ylabel("score")
    a.set_title("(A) Number of operators", fontsize=9.5)
    a.legend(fontsize=7.5, frameon=False)
    b = ax[0, 1]
    ns = list(plen)
    line(b, ns, plen, "modelid_auc", C_MID, "o", "model-ID AUC")
    line(b, ns, plen, "correct_wpauc", C_WP, "s", "correctness WP-AUC")
    b.axvline(3, color="grey", ls=":", lw=1)
    b.set_xticks(ns)
    b.set_xlabel("pivot length (tokens)")
    b.set_title("(B) Pivot length", fontsize=9.5)
    b.legend(fontsize=7.5, frameon=False)
    c = ax[1, 0]
    fs = list(pfreq)
    line(c, fs, pfreq, "modelid_auc", C_MID, "o", "model-ID AUC")
    line(c, fs, pfreq, "correct_wpauc", C_WP, "s", "correctness WP-AUC")
    c.axvline(100, color="grey", ls=":", lw=1)
    c.set_xscale("log")
    c.set_xticks(fs)
    c.xaxis.set_major_formatter(ScalarFormatter())
    c.xaxis.set_minor_locator(plt.NullLocator())
    c.set_xlabel("min-trace threshold")
    c.set_ylabel("score")
    c.set_title("(C) Frequency threshold", fontsize=9.5)
    c.legend(fontsize=7.5, frameon=False)
    d = ax[1, 1]
    if joint:
        for j, (key, color, label) in enumerate(
            [("modelid_auc", C_MID, "model-ID AUC"), ("correct_wpauc", C_WP, "correctness WP-AUC")]
        ):
            ys = [r[key] for r in joint]
            xs = [j + (i / max(len(ys) - 1, 1) - 0.5) * 0.35 for i in range(len(ys))]
            d.scatter(
                xs, sorted(ys), s=16, color=color, alpha=0.75, edgecolors="white", linewidths=0.5
            )
            if default_rows:
                dm = st.mean([r[key] for r in default_rows])
                d.plot([j - 0.28, j + 0.28], [dm, dm], color=color, lw=1.6)
                d.annotate("default", (j + 0.3, dm), fontsize=7, color=color, va="center")
        d.set_xticks([0, 1])
        d.set_xticklabels(["model-ID AUC", "correctness WP-AUC"])
        d.set_xlim(-0.5, 1.7)
        d.set_title(f"(D) joint perturbation ({len(joint)} random configs)", fontsize=9.5)
        d.set_ylabel("score")
    fig.suptitle("Hyperparameter sensitivity of the operator pipeline", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"robustness.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {RES / 'robustness.png'} / .pdf")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
