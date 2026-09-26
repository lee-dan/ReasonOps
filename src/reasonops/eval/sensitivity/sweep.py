"""Evaluate sensitivity to operator-discovery parameters."""

from reasonops.paths import DATA_DIR, OUTPUT_DIR

HERE = OUTPUT_DIR / "sensitivity"
import argparse
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, roc_auc_score, silhouette_score

from reasonops.utils import (
    EXCLUDE_DATASETS,
    MODEL_ORDER,
    N_FOLDS,
    SEED,
    problem_kfold_splits,
    wp_auc,
)

TRACE_CACHE = HERE / "cache/trace_cache.pkl"
START_CACHE = HERE / "cache/start_cache.pkl"
EMB_DEFAULT = DATA_DIR / "k_sweep/ngram_embeddings.npy"
OUT = HERE / "results/sweep.jsonl"
DEFAULT = dict(n=3, freq=100, dom=3, vocab=2000)
EMBEDDERS = {
    "e5": "intfloat/e5-small-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}
TRACE_AXES = ("k", "algo", "embedder")
START_AXES = ("length", "freq", "domain", "vocab", "joint")


def op_features(seq, K):
    """Generalize the 117-feature operator representation to K clusters."""
    n = len(seq)
    counts = np.zeros(K, dtype=np.float32)
    for op in seq:
        if 0 <= op < K:
            counts[op] += 1
    freq = counts / max(n, 1)
    q_size = max(1, n // 4)
    quart = []
    for q in range(4):
        s = q * q_size
        e = (q + 1) * q_size if q < 3 else n
        qc = np.zeros(K, dtype=np.float32)
        for op in seq[s:e]:
            if 0 <= op < K:
                qc[op] += 1
        quart.append(qc / max(e - s, 1))
    nz = freq[freq > 0]
    entropy = float(-np.sum(nz * np.log(nz + 1e-12))) if len(nz) else 0.0
    scalars = np.array(
        [
            entropy,
            float(freq.max()) if n > 0 else 0.0,
            float((freq > 0).sum()),
            float(np.log1p(n)),
            float(sum((seq[i] == seq[i - 1] for i in range(1, n)))) / max(n - 1, 1),
        ],
        dtype=np.float32,
    )
    first_oh = np.zeros(K, dtype=np.float32)
    last_oh = np.zeros(K, dtype=np.float32)
    if n > 0 and 0 <= seq[0] < K:
        first_oh[seq[0]] = 1.0
    if n > 0 and 0 <= seq[-1] < K:
        last_oh[seq[-1]] = 1.0
    bigrams = np.zeros(K * K, dtype=np.float32)
    for i in range(1, n):
        a, b = (seq[i - 1], seq[i])
        if 0 <= a < K and 0 <= b < K:
            bigrams[a * K + b] += 1
    if n > 1:
        bigrams /= n - 1
    run_mean = np.zeros(K, dtype=np.float32)
    run_max = np.zeros(K, dtype=np.float32)
    runs = {k: [] for k in range(K)}
    if n > 0:
        cur, cl = (seq[0], 1)
        for op in seq[1:]:
            if op == cur:
                cl += 1
            else:
                if 0 <= cur < K:
                    runs[cur].append(cl)
                cur, cl = (op, 1)
        if 0 <= cur < K:
            runs[cur].append(cl)
        for k in range(K):
            if runs[k]:
                run_mean[k] = float(np.mean(runs[k])) / max(n, 1)
                run_max[k] = float(max(runs[k])) / max(n, 1)
    return np.concatenate([freq, *quart, scalars, first_oh, last_oh, bigrams, run_mean, run_max])


def make_clf(n_classes, seed, quick):
    from xgboost import XGBClassifier

    kw = dict(
        n_estimators=60 if quick else 400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    if n_classes == 2:
        return XGBClassifier(eval_metric="logloss", **kw)
    return XGBClassifier(
        objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **kw
    )


def eval_modelid(X, mi, pids, seed, quick):
    m = mi >= 0
    X, mi, pids = (X[m], mi[m], pids[m])
    if quick:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=min(6000, len(X)), replace=False)
        X, mi, pids = (X[keep], mi[keep], pids[keep])
    classes, encoded = np.unique(mi, return_inverse=True)
    N = len(mi)
    proba = np.zeros((N, len(classes)))
    pred = np.full(N, -1)
    for tr, te in problem_kfold_splits(pids, N_FOLDS, seed):
        clf = make_clf(len(classes), seed, quick)
        clf.fit(X[tr], encoded[tr])
        p = clf.predict_proba(X[te])
        proba[te] = p
        pred[te] = classes[np.argmax(p, axis=1)]
    Yb = np.zeros((N, len(classes)))
    for j, c in enumerate(classes):
        Yb[mi == c, j] = 1
    auc = roc_auc_score(Yb, proba, average="macro", multi_class="ovr")
    return (float(auc), float(accuracy_score(mi, pred)))


def eval_correct(X, yc, pids, seed, quick):
    m = yc >= 0
    X, yc, pids = (X[m], yc[m], pids[m])
    if quick:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=min(6000, len(X)), replace=False)
        X, yc, pids = (X[keep], yc[keep], pids[keep])
    N = len(yc)
    oof = np.full(N, np.nan)
    for tr, te in problem_kfold_splits(pids, N_FOLDS, seed):
        if len(set(yc[tr].tolist())) < 2:
            continue
        clf = make_clf(2, seed, quick)
        clf.fit(X[tr], yc[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    v = ~np.isnan(oof)
    return float(wp_auc(yc[v], oof[v], pids[v]))


def cluster(emb, algo, K, seed):
    if algo == "kmeans":
        return KMeans(n_clusters=K, random_state=seed, n_init=30, max_iter=500).fit_predict(emb)
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(emb)
    if algo == "hdbscan":
        from sklearn.cluster import HDBSCAN

        lab = HDBSCAN(min_cluster_size=50, min_samples=10).fit_predict(emb)
        uniq = sorted(set(lab.tolist()))
        remap = {c: i for i, c in enumerate(uniq)}
        return np.array([remap[c] for c in lab], dtype=int)
    raise ValueError(algo)


_emb_models = {}
_emb_cache = {}


def embed(phrases, embedder="e5"):
    todo = [ph for ph in phrases if (embedder, ph) not in _emb_cache]
    if todo:
        if embedder not in _emb_models:
            from sentence_transformers import SentenceTransformer

            _emb_models[embedder] = SentenceTransformer(EMBEDDERS[embedder])
        vecs = (
            _emb_models[embedder]
            .encode(todo, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            .astype(np.float64)
        )
        for ph, v in zip(todo, vecs):
            _emb_cache[embedder, ph] = v
    return np.array([_emb_cache[embedder, ph] for ph in phrases])


def build_matrices(cache, ng_labels, K):
    model_to_i = {m: i for i, m in enumerate(MODEL_ORDER)}
    X, yc, mi, pids, ds = ([], [], [], [], [])
    for t in cache["traces"].values():
        seq = ng_labels[t["pivot_idx"]]
        if len(seq) == 0:
            continue
        X.append(op_features(seq.tolist(), K))
        c = t["correct"]
        yc.append(-1 if c is None else int(bool(c)))
        mi.append(model_to_i.get(t["model"], -1))
        pids.append(t["problem_id"])
        ds.append(t["dataset"])
    X = np.asarray(X, dtype=np.float32)
    ds = np.asarray(ds)
    mask6 = ~np.isin(ds, list(EXCLUDE_DATASETS))
    return (
        X,
        np.asarray(yc, dtype=np.int8),
        np.asarray(mi, dtype=np.int16),
        np.asarray(pids),
        mask6,
    )


def trace_cells(axes, seeds):
    cells = []
    if "k" in axes:
        for K in range(4, 13):
            for s in seeds:
                cells.append(
                    dict(axis="k", tag=f"k={K}", algo="kmeans", K=K, embedder="e5", seed=s)
                )
    if "algo" in axes:
        for algo in ("kmeans", "agglomerative", "hdbscan"):
            for s in seeds if algo == "kmeans" else seeds[:1]:
                cells.append(
                    dict(axis="algo", tag=f"algo={algo}", algo=algo, K=7, embedder="e5", seed=s)
                )
    if "embedder" in axes:
        for e in EMBEDDERS:
            for s in seeds:
                cells.append(
                    dict(axis="embedder", tag=f"emb={e}", algo="kmeans", K=7, embedder=e, seed=s)
                )
    return cells


def run_trace_axes(axes, seeds, quick, fout):
    cache = pickle.loads(TRACE_CACHE.read_bytes())
    ng_list = cache["ngram_list"]
    phrases = [" ".join(ng) if isinstance(ng, (list, tuple)) else ng for ng in ng_list]
    if EMB_DEFAULT.exists():
        emb_e5 = np.load(EMB_DEFAULT)
    else:
        emb_e5 = embed(phrases)
        EMB_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMB_DEFAULT, emb_e5)
    assert emb_e5.shape[0] == len(ng_list)
    _emb_cache.update({("e5", ph): v for ph, v in zip(phrases, emb_e5)})
    ref_labels = cluster(emb_e5, "kmeans", 7, SEED)
    for cell in trace_cells(axes, seeds):
        t0 = time.time()
        emb = embed(phrases, cell["embedder"])
        labels = cluster(emb, cell["algo"], cell["K"], cell["seed"])
        Kc = int(labels.max()) + 1
        n_clusters = int(len(set(labels.tolist())))
        sil = float(silhouette_score(emb, labels)) if n_clusters > 1 else float("nan")
        ari = float(adjusted_rand_score(ref_labels, labels))
        X, yc, mi, pids, mask6 = build_matrices(cache, labels, Kc)
        mid_auc, mid_acc = eval_modelid(X, mi, pids, SEED, quick)
        cwp = eval_correct(X[mask6], yc[mask6], pids[mask6], SEED, quick)
        emit(
            fout,
            dict(
                axis=cell["axis"],
                tag=cell["tag"],
                seed=cell["seed"],
                algo=cell["algo"],
                K_req=cell["K"],
                embedder=cell["embedder"],
                n_pivots=len(ng_list),
                n_clusters=n_clusters,
                silhouette=round(sil, 4),
                ari_vs_default=round(ari, 4),
                modelid_auc=round(mid_auc, 4),
                modelid_acc=round(mid_acc, 4),
                correct_wpauc=round(cwp, 4),
                secs=round(time.time() - t0, 1),
                quick=quick,
            ),
        )


def accepted_pivots(traces, n, freq, dom, top_vocab):
    from collections import defaultdict

    ntraces = defaultdict(int)
    ndoms = defaultdict(set)
    for t in traces.values():
        seen = set()
        for st in t["starts"]:
            p = st[:n]
            if p and p not in seen:
                seen.add(p)
                ntraces[p] += 1
                ndoms[p].add(t["dataset"])
    return {
        p
        for p, c in ntraces.items()
        if c >= freq and len(ndoms[p]) >= dom and all((tok in top_vocab for tok in p))
    }


def start_matrices(traces, n, accepted, piv2cluster, K=7):
    m2i = {m: i for i, m in enumerate(MODEL_ORDER)}
    X, yc, mi, pids, ds = ([], [], [], [], [])
    for t in traces.values():
        seq = [piv2cluster[st[:n]] for st in t["starts"] if st[:n] in accepted]
        if not seq:
            continue
        X.append(op_features(seq, K))
        c = t["correct"]
        yc.append(-1 if c is None else int(c))
        mi.append(m2i.get(t["model"], -1))
        pids.append(t["problem_id"])
        ds.append(t["dataset"])
    X = np.asarray(X, dtype=np.float32)
    ds = np.asarray(ds)
    mask6 = ~np.isin(ds, list(EXCLUDE_DATASETS))
    return (
        X,
        np.asarray(yc, dtype=np.int8),
        np.asarray(mi, dtype=np.int16),
        np.asarray(pids),
        mask6,
    )


def start_cells(axes, seeds):
    cells, seen = ([], set())

    def add(axis, seed_list, **kw):
        c = dict(DEFAULT)
        c.update(kw)
        for s in seed_list:
            key = (axis, tuple(sorted(c.items())), s)
            if key not in seen:
                seen.add(key)
                cells.append(dict(axis=axis, seed=s, **c))

    add("default", seeds)
    if "length" in axes:
        for n in (1, 2, 4, 5):
            add("length", seeds, n=n)
    if "freq" in axes:
        for f in (25, 50, 200, 400):
            add("freq", seeds, freq=f)
    if "domain" in axes:
        for d in (1, 2, 4, 5):
            add("domain", seeds, dom=d)
    if "vocab" in axes:
        for v in (1000, 1500, 3000, 4000):
            add("vocab", seeds, vocab=v)
    if "joint" in axes:
        grid = list(
            itertools.product(
                (2, 3, 4), (25, 50, 100, 200, 400), (1, 2, 3, 4, 5), (1000, 1500, 2000, 3000, 4000)
            )
        )
        grid = [g for g in grid if dict(zip(("n", "freq", "dom", "vocab"), g)) != DEFAULT]
        rng = np.random.default_rng(0)
        for i in rng.choice(len(grid), size=24, replace=False):
            g = grid[i]
            add("joint", seeds[:1], n=g[0], freq=g[1], dom=g[2], vocab=g[3])
    return cells


def run_start_axes(axes, seeds, quick, fout):
    cache = pickle.loads(START_CACHE.read_bytes())
    traces = cache["traces"]
    top_vocab_ranked = cache["top_vocab_ranked"]
    print(f"start cache: {len(traces):,} traces", flush=True)
    default_ref = None
    for cell in start_cells(axes, seeds):
        t0 = time.time()
        n, freq, dom, vocab, seed = (
            cell["n"],
            cell["freq"],
            cell["dom"],
            cell["vocab"],
            cell["seed"],
        )
        tag = f"n={n},f={freq},d={dom},v={vocab}"
        acc = accepted_pivots(traces, n, freq, dom, set(top_vocab_ranked[:vocab]))
        piv_sorted = sorted(acc)
        if len(piv_sorted) < 7:
            emit(
                fout,
                dict(
                    axis=cell["axis"],
                    tag=tag,
                    seed=seed,
                    **{k: cell[k] for k in ("n", "freq", "dom", "vocab")},
                    n_pivots=len(piv_sorted),
                    error="too_few_pivots",
                ),
            )
            continue
        emb = embed([" ".join(p) for p in piv_sorted])
        labels = cluster(emb, "kmeans", 7, seed)
        piv2cluster = {p: int(l) for p, l in zip(piv_sorted, labels)}
        sil = float(silhouette_score(emb, labels))
        ari = float("nan")
        if default_ref is not None and n == DEFAULT["n"]:
            shared = [p for p in piv_sorted if p in default_ref]
            if len(shared) > 10:
                ari = float(
                    adjusted_rand_score(
                        [default_ref[p] for p in shared], [piv2cluster[p] for p in shared]
                    )
                )
        if default_ref is None and cell["axis"] == "default":
            default_ref = piv2cluster
        X, yc, mi, pids, mask6 = start_matrices(traces, n, acc, piv2cluster)
        mid_auc, mid_acc = eval_modelid(X, mi, pids, SEED, quick)
        cwp = eval_correct(X[mask6], yc[mask6], pids[mask6], SEED, quick)
        emit(
            fout,
            dict(
                axis=cell["axis"],
                tag=tag,
                seed=seed,
                n=n,
                freq=freq,
                dom=dom,
                vocab=vocab,
                n_pivots=len(piv_sorted),
                n_clusters=int(len(set(labels))),
                silhouette=round(sil, 4),
                ari_vs_default=round(ari, 4),
                modelid_auc=round(mid_auc, 4),
                modelid_acc=round(mid_acc, 4),
                correct_wpauc=round(cwp, 4),
                secs=round(time.time() - t0, 1),
                quick=quick,
            ),
        )


def emit(fout, row):
    fout.write(json.dumps(row) + "\n")
    fout.flush()
    print(
        f"[{row['axis']:8s}|{row['tag']:24s}|s{row['seed']}] "
        + " ".join(
            (
                f"{k}={row[k]}"
                for k in (
                    "n_pivots",
                    "silhouette",
                    "ari_vs_default",
                    "modelid_auc",
                    "correct_wpauc",
                    "secs",
                )
                if k in row
            )
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="+", default=["all"], choices=["all", *TRACE_AXES, *START_AXES])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1, 2],
        help="kmeans-init seeds (first one is the ARI reference)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="output jsonl (give parallel jobs separate files; analyze.py reads every results/sweep*.jsonl)",
    )
    args = ap.parse_args()
    axes = TRACE_AXES + START_AXES if "all" in args.axis else tuple(args.axis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fout:
        if any((a in TRACE_AXES for a in axes)):
            run_trace_axes(axes, args.seeds, args.quick, fout)
        if any((a in START_AXES for a in axes)):
            run_start_axes(axes, args.seeds, args.quick, fout)
    print(f"\ndone -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
