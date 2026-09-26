"""Evaluate model identification under persona, evasion, and mimicry prompts."""

from reasonops.operators.pivot_dictionary import load_pivots
from reasonops.paths import DATA_DIR, OUTPUT_DIR

HERE = OUTPUT_DIR / "system_prompting"
import asyncio
import gzip
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from reasonops.data.benchmarks import load_dataset_problems
from reasonops.data.grade import get_verifier
from reasonops.data.models import MODEL_REGISTRY, PROMPT_TEMPLATES
from reasonops.operators.discover_operators import segment_traces, sentence_start, split_sentences

TRACES = HERE / "traces"
RESULTS = HERE / "results"
CORPUS = DATA_DIR / "final_dataset.jsonl.gz"
NGRAMS = DATA_DIR / "k_sweep/ngram_list.json"
SIG = RESULTS / "signature_pivots.json"
CLF_CACHE = RESULTS / "clf_cache.pkl"
MODELS = ["qwen3-30b-thinking", "kimi-k2.5", "r1-0528", "r1-distill-llama-70b"]
GEN_MODELS = MODELS[:3]
MIMIC_TARGET = {
    "qwen3-30b-thinking": "kimi-k2.5",
    "kimi-k2.5": "r1-0528",
    "r1-0528": "qwen3-30b-thinking",
    "r1-distill-llama-70b": "kimi-k2.5",
}
DATASETS = {"aime": 30, "gpqa": 40}
MAX_TOKENS = 24000
CONCURRENCY = 96
GENURL = "https://openrouter.ai/api/v1/chat/completions"
BENIGN = {
    "expert": "You are a world-class expert with deep mastery of this subject.",
    "stakes": "This is a high-stakes problem where accuracy is critical.",
    "careful": "You are an exceptionally careful and self-critical thinker.",
}
EVADE_BLIND = "Your reasoning will be analyzed by a classifier that identifies which AI model produced it. Make your reasoning style as unrecognizable as possible: avoid your habitual phrasing and stylistic tics. Still solve the problem correctly."
EVADE_METHOD = "Your reasoning will be analyzed by a classifier that identifies which AI model produced it from the first few words of each sentence (discourse markers such as 'Wait', 'So', 'Alternatively', 'Let me check'). Vary your sentence openers and avoid your habitual discourse markers. Still solve the problem correctly."
MIMIC = "Your reasoning will be analyzed by a classifier that identifies which AI model produced it from the first few words of each sentence. Evade it by imitating a different model's reasoning style: where natural, begin your sentences with openers such as: {openers}. Avoid your own habitual openers. Still solve the problem correctly."


def framing_text(framing, model):
    if framing == "neutral":
        return ""
    if framing in BENIGN:
        return BENIGN[framing]
    if framing == "evade_blind":
        return EVADE_BLIND
    if framing == "evade_method":
        return EVADE_METHOD
    if framing == "mimic":
        sig = json.loads(SIG.read_text())
        openers = "; ".join((f"'{p}'" for p in sig[MIMIC_TARGET[model]][:12]))
        return MIMIC.format(openers=openers)
    raise KeyError(framing)


FRAMINGS = ["neutral"] + list(BENIGN) + ["evade_blind", "evade_method", "mimic"]


def pivots():
    type_a = frozenset((tuple(g.split()) for g in load_pivots()))
    need = set(MIMIC_TARGET.values())
    counts, totals = (defaultdict(Counter), Counter())
    with gzip.open(CORPUS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            m = r["model"]
            for sp in r.get("spans", []):
                st = sentence_start(sp.get("anchor", ""), n=3)
                if st in type_a:
                    totals[m] += 1
                    counts[m][" ".join(st)] += 1
    out = {}
    for m in need:
        n_m = totals[m]
        n_rest = sum(totals.values()) - n_m
        scored = []
        for g, c in counts[m].items():
            if c < 50:
                continue
            c_rest = sum((counts[mm][g] for mm in counts if mm != m))
            lo = np.log((c + 1) / n_m) - np.log((c_rest + 1) / max(n_rest, 1))
            scored.append((lo, g))
        out[m] = [g for _, g in sorted(scored, reverse=True)[:15]]
        print(f"{m}: {out[m][:8]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    SIG.write_text(json.dumps(out, indent=2))
    print(f"wrote {SIG}")


async def one(session, sem, key, model, framing, ds, prob, verify, tmpl):
    out_path = TRACES / model / framing / ds / f"{str(prob['problem_id']).replace('/', '_')}.json"
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = MODEL_REGISTRY[model]
    rf = cfg["reasoning_format"]
    reasoning = (
        {"effort": "high"}
        if rf == "effort"
        else {"max_tokens": MAX_TOKENS - 2000}
        if rf == "max_tokens"
        else {}
    )
    sys_text = framing_text(framing, model)
    messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + [
        {"role": "user", "content": tmpl.format(problem=prob["problem"])}
    ]
    payload = {
        "model": cfg["or_model_id"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "reasoning": reasoning,
        "usage": {"include": True},
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await session.post(
                    GENURL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=500,
                )
                r.raise_for_status()
                d = r.json()
                ch = d["choices"][0]["message"]
                u = d.get("usage", {})
                text = (ch.get("content") or "") or (ch.get("reasoning") or "")
                rec = {
                    "model": model,
                    "framing": framing,
                    "dataset": ds,
                    "problem_id": prob["problem_id"],
                    "reasoning": ch.get("reasoning") or "",
                    "response": ch.get("content") or "",
                    "correct": bool(verify(text, prob)) if text else None,
                    "completion_tokens": u.get("completion_tokens"),
                    "cost": u.get("cost"),
                }
                out_path.write_text(json.dumps(rec, ensure_ascii=False))
                return
            except Exception as e:
                if attempt == 3:
                    print(f"  FAIL {model}/{framing}/{ds}/{prob['problem_id']}: {e}", flush=True)
                else:
                    await asyncio.sleep(2**attempt)


async def gen():
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(CONCURRENCY)
    async with __import__("httpx").AsyncClient() as session:
        tasks = []
        for ds, n in DATASETS.items():
            probs = load_dataset_problems(ds, n=n)
            verify = get_verifier(ds)
            tmpl = PROMPT_TEMPLATES[ds]
            for m in GEN_MODELS:
                for fr in FRAMINGS:
                    for p in probs:
                        tasks.append(one(session, sem, key, m, fr, ds, p, verify, tmpl))
        print(f"{len(tasks)} generation tasks", flush=True)
        await asyncio.gather(*tasks)
    print("gen done", flush=True)


def featurize(texts):
    piv2c = load_pivots()
    type_a = frozenset((tuple(g.split()) for g in piv2c))
    sents = {i: split_sentences(t or "") for i, t in enumerate(texts)}
    spans_by_id = segment_traces(sents, type_a)
    seqs, anchors = ([], [])
    for i in range(len(texts)):
        seq, parts = ([], [])
        for sp in spans_by_id[i]:
            st = sentence_start(sp["anchor"], n=3)
            c = piv2c.get(" ".join(st), -1) if st else -1
            if c >= 0:
                seq.append(c)
            parts.append(sp["anchor"][:80])
        seqs.append(seq)
        anchors.append(" ".join(parts) if parts else (texts[i] or "")[-3000:])
    return (seqs, anchors)


def train_frozen():
    RESULTS.mkdir(parents=True, exist_ok=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    from reasonops.prediction.op_seq_baseline import op_features
    from reasonops.utils import EXCLUDE_DATASETS, SEED, make_multiclass_clf

    if CLF_CACHE.exists():
        return pickle.loads(CLF_CACHE.read_bytes())
    models, seqs, anchors = ([], [], [])
    with gzip.open(CORPUS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("dataset", "") in EXCLUDE_DATASETS or not r.get("model"):
                continue
            seq = [int(x) for x in r.get("operator_sequence", [])]
            if not seq:
                continue
            models.append(r["model"])
            seqs.append(seq)
            parts = [sp.get("anchor", "")[:80] for sp in r.get("spans", []) if sp.get("anchor")]
            resp = r.get("model_response") or r.get("reasoning") or ""
            anchors.append(" ".join(parts) if parts else resp[-3000:])
    le = LabelEncoder()
    y = le.fit_transform(models)
    print(f"corpus: {len(y):,} traces, classes {list(le.classes_)}", flush=True)
    X_op = np.array([op_features(s) for s in seqs], dtype=np.float32)
    tfidf = TfidfVectorizer(max_features=8000, sublinear_tf=True)
    T = tfidf.fit_transform(anchors).toarray().astype(np.float32)
    full = make_multiclass_clf(len(le.classes_), SEED)
    full.fit(np.hstack([T, X_op]), y)
    print("full classifier trained", flush=True)
    op_only = make_multiclass_clf(len(le.classes_), SEED)
    op_only.fit(X_op, y)
    print("op-only classifier trained", flush=True)
    bundle = {"full": full, "op_only": op_only, "tfidf": tfidf, "labels": list(le.classes_)}
    CLF_CACHE.write_bytes(pickle.dumps(bundle))
    return bundle


def analyze():
    from reasonops.prediction.op_seq_baseline import op_features

    bundle = train_frozen()
    labels = bundle["labels"]
    li = {m: j for j, m in enumerate(labels)}
    recs = [json.loads(f.read_text()) for f in sorted(TRACES.rglob("*.json"))]
    recs = [r for r in recs if r.get("reasoning")]
    seqs, anchors = featurize([r["reasoning"] for r in recs])
    keep = [i for i, s in enumerate(seqs) if s]
    recs = [recs[i] for i in keep]
    seqs = [seqs[i] for i in keep]
    X_op = np.array([op_features(s) for s in seqs], dtype=np.float32)
    T = bundle["tfidf"].transform([anchors[i] for i in keep]).toarray().astype(np.float32)
    P_full = bundle["full"].predict_proba(np.hstack([T, X_op]))
    P_op = bundle["op_only"].predict_proba(X_op)
    agg = defaultdict(list)
    for i, r in enumerate(recs):
        agg[r["model"], r["framing"]].append(i)
    report = [
        "# Model identification under system prompts\n",
        "Frozen 12-way Op-XGB classifiers trained on the paper corpus (chance = 8.3%).",
        "`full` = TF-IDF + operator features (paper config); `op` = 117-dim operator",
        "features only. P(self) = mean probability on the true model. `mimic` rows",
        "also report P(target) and %->target for the imitated model.\n",
        "| model | framing | n | ID% full | ID% op | P(self) | P(target) | %->target | task acc |",
        "|--|--|--|--|--|--|--|--|--|",
    ]
    for m in MODELS:
        for fr in FRAMINGS:
            idx = agg.get((m, fr))
            if not idx:
                continue
            j = li[m]
            id_full = float(np.mean(P_full[idx].argmax(1) == j))
            id_op = float(np.mean(P_op[idx].argmax(1) == j))
            p_self = float(np.mean(P_full[idx][:, j]))
            accs = [recs[i]["correct"] for i in idx if recs[i]["correct"] is not None]
            acc = float(np.mean(accs)) if accs else float("nan")
            if fr == "mimic":
                jt = li[MIMIC_TARGET[m]]
                p_t = f"{float(np.mean(P_full[idx][:, jt])):.3f}"
                to_t = f"{100 * float(np.mean(P_full[idx].argmax(1) == jt)):.0f}"
            else:
                p_t, to_t = ("—", "—")
            report.append(
                f"| {m} | {fr} | {len(idx)} | {100 * id_full:.0f} | {100 * id_op:.0f} | {p_self:.3f} | {p_t} | {to_t} | {acc:.3f} |"
            )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    dists = {
        f"{m}|{fr}": np.mean(
            [np.bincount(seqs[i], minlength=7)[:7] / max(len(seqs[i]), 1) for i in idx], axis=0
        )
        .round(4)
        .tolist()
        for (m, fr), idx in agg.items()
    }
    (RESULTS / "op_dists.json").write_text(json.dumps(dists, indent=2))
    print(f"\nwrote {RESULTS / 'results.md'} and op_dists.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pivots", "gen", "analyze"])
    args = parser.parse_args()
    {"pivots": pivots, "gen": lambda: asyncio.run(gen()), "analyze": analyze}[args.mode]()
