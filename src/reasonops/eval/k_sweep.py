#!/usr/bin/env python3
"""
K selection sweep: run KMeans for K in {6..11} on the pivot ngrams and
compute LLM-judge κ (Claude Sonnet 4.6) for each K to identify the optimal K.

Outputs:
  data/k_sweep/kappa_k{K}.json   — per-K kappa results
  data/k_sweep/summary.json      — all K values + kappa in one table

Usage:
    python -m reasonops.eval.k_sweep \\
        --spans data/operators/discovered_k7/spans_clustered.jsonl \\
        --out   data/k_sweep

Requires: ANTHROPIC_API_KEY
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score


# ── Pivot embedding ──────────────────────────────────────────────────────────

def embed_ngrams(ngrams: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-small-v2")
    vecs = model.encode(ngrams, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
    return vecs.astype(np.float64)


# ── Cluster naming ───────────────────────────────────────────────────────────

def name_cluster(client, cluster_id: int, spans: list[dict],
                 top_ngrams: list[str], judge_model: str,
                 n_exemplars: int = 5, seed: int = 42) -> dict:
    rng = random.Random(seed + cluster_id)
    examples = rng.sample(spans, min(n_exemplars, len(spans)))
    example_texts = [s["anchor"][:200] for s in examples]
    prompt = (
        f"Below are the most common opening phrases for a cluster of reasoning spans, "
        f"followed by {len(example_texts)} example spans.\n\n"
        f"Top opening phrases:\n" +
        "\n".join(f"  {ng}" for ng in top_ngrams[:12]) +
        f"\n\nExample spans:\n" +
        "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(example_texts)) +
        "\n\nIn one sentence, name and describe the reasoning move this cluster represents. "
        "Be concise and specific."
    )
    resp = client.messages.create(
        model=judge_model,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    name = resp.content[0].text.strip()
    return {"cluster_id": cluster_id, "name": name, "top_ngrams": top_ngrams}


# ── Judge validation ─────────────────────────────────────────────────────────

def classify_span(client, span: dict, names_by_id: dict, K: int,
                  judge_model: str = "claude-sonnet-4-6") -> int:
    cats_lines = []
    for cid in range(K):
        n = names_by_id.get(cid, {}).get("name", f"Cluster {cid}")
        ngs = names_by_id.get(cid, {}).get("top_ngrams", [])
        line = f"  {cid+1}. {n}"
        if ngs:
            line += f"\n     Typical openers: {', '.join(repr(ng) for ng in ngs[:8])}"
        cats_lines.append(line)

    prompt = (
        "You are classifying a sentence from an LLM reasoning trace.\n\n"
        "Categories:\n" + "\n".join(cats_lines) +
        f"\n\nSpan: \"{span['anchor'][:150]}\"\n\n"
        f"Respond with ONLY the category number (1–{K}). No explanation."
    )
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=judge_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            if not resp.content:
                continue
            text = resp.content[0].text.strip()
            m = re.search(r"\d+", text)
            if m:
                return int(m.group()) - 1
        except Exception:
            pass
    return 0


def compute_kappa(client, spans_by_cluster: dict, names_by_id: dict, K: int,
                  judge_model: str = "claude-sonnet-4-6",
                  n_per_cluster: int = 50, seed: int = 42) -> dict:
    rng = random.Random(seed)
    sample = []
    for cid, spans in spans_by_cluster.items():
        chosen = rng.sample(spans, min(n_per_cluster, len(spans)))
        for sp in chosen:
            sample.append({"span": sp, "true_cluster": cid})

    print(f"  Classifying {len(sample)} spans with {judge_model}...")
    y_true, y_pred = [], []
    for i, item in enumerate(sample):
        if i % 50 == 0:
            print(f"    {i}/{len(sample)}", flush=True)
        pred = classify_span(client, item["span"], names_by_id, K, judge_model=judge_model)
        y_true.append(item["true_cluster"])
        y_pred.append(pred)

    kappa = float(cohen_kappa_score(y_true, y_pred))
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    per_cluster = {}
    for cid in range(K):
        idxs = [i for i, t in enumerate(y_true) if t == cid]
        if idxs:
            acc_c = sum(1 for i in idxs if y_pred[i] == cid) / len(idxs)
            per_cluster[cid] = {"n": len(idxs), "accuracy": round(acc_c, 4)}

    return {
        "kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "n_sample": len(sample),
        "per_cluster": per_cluster,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True,
                    help="spans_clustered.jsonl from discovered_k7")
    ap.add_argument("--out",          default="data/k_sweep")
    ap.add_argument("--k_values",     nargs="+", type=int, default=None,
                    help="Explicit K values to sweep (overrides --k_min/--k_max)")
    ap.add_argument("--k_min",        type=int, default=6)
    ap.add_argument("--k_max",        type=int, default=11)
    ap.add_argument("--judge_model",  default="claude-sonnet-4-6")
    ap.add_argument("--n_per_cluster",type=int, default=50)
    ap.add_argument("--n_exemplars",  type=int, default=5)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    if args.k_values is None:
        args.k_values = list(range(args.k_min, args.k_max + 1))

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load spans
    print("Loading spans...", flush=True)
    all_spans = []
    with open(args.spans) as f:
        for line in f:
            all_spans.append(json.loads(line))
    print(f"  {len(all_spans):,} spans", flush=True)

    # Extract unique ngrams and their spans
    ngram_spans = defaultdict(list)
    for sp in all_spans:
        ng = sp.get("ngram")
        if ng:
            ngram_spans[ng].append(sp)

    unique_ngrams = sorted(ngram_spans.keys())
    print(f"  {len(unique_ngrams):,} unique ngrams", flush=True)

    # Embed ngrams (cached)
    emb_cache = out_dir / "ngram_embeddings.npy"
    ngram_list_cache = out_dir / "ngram_list.json"

    if emb_cache.exists() and ngram_list_cache.exists():
        print("Loading cached embeddings...", flush=True)
        cached_ngrams = json.loads(ngram_list_cache.read_text())
        if cached_ngrams == unique_ngrams:
            embeddings = np.load(emb_cache)
            print(f"  Loaded {embeddings.shape}", flush=True)
        else:
            print("  Cache mismatch, re-embedding...", flush=True)
            embeddings = embed_ngrams(unique_ngrams)
            np.save(emb_cache, embeddings)
            ngram_list_cache.write_text(json.dumps(unique_ngrams))
    else:
        print("Embedding ngrams...", flush=True)
        embeddings = embed_ngrams(unique_ngrams)
        np.save(emb_cache, embeddings)
        ngram_list_cache.write_text(json.dumps(unique_ngrams))
        print(f"  Saved embeddings: {embeddings.shape}", flush=True)

    summary = {}

    for K in args.k_values:
        print(f"\n{'='*60}", flush=True)
        print(f"K = {K}", flush=True)

        out_path = out_dir / f"kappa_k{K}.json"
        if out_path.exists():
            print(f"  Already exists: {out_path}", flush=True)
            result = json.loads(out_path.read_text())
            summary[K] = result.get("kappa")
            continue

        # Cluster ngrams
        print(f"  KMeans k={K}...", flush=True)
        km = KMeans(n_clusters=K, random_state=args.seed, n_init=30, max_iter=500)
        ng_labels = km.fit_predict(embeddings)
        ngram_to_cluster = {ng: int(lab) for ng, lab in zip(unique_ngrams, ng_labels)}

        # Re-label spans
        spans_by_cluster = defaultdict(list)
        for sp in all_spans:
            ng = sp.get("ngram")
            if ng and ng in ngram_to_cluster:
                spans_by_cluster[ngram_to_cluster[ng]].append(sp)

        cluster_sizes = {c: len(ss) for c, ss in spans_by_cluster.items()}
        print(f"  Cluster sizes: {sorted(cluster_sizes.values(), reverse=True)}", flush=True)

        # Top ngrams per cluster
        top_ngrams_by_cluster = {}
        for cid in range(K):
            ctr = Counter(sp["ngram"] for sp in spans_by_cluster[cid] if sp.get("ngram"))
            top_ngrams_by_cluster[cid] = [ng for ng, _ in ctr.most_common(12)]

        # Name clusters
        print(f"  Naming {K} clusters...", flush=True)
        names_by_id = {}
        for cid in range(K):
            print(f"    Naming cluster {cid}...", flush=True)
            names_by_id[cid] = name_cluster(
                client, cid, spans_by_cluster[cid],
                top_ngrams_by_cluster[cid],
                judge_model=args.judge_model,
                n_exemplars=args.n_exemplars,
                seed=args.seed,
            )
            print(f"      → {names_by_id[cid]['name']}", flush=True)

        # Validate
        print(f"  Running judge validation (n_per_cluster={args.n_per_cluster})...", flush=True)
        kappa_result = compute_kappa(
            client, spans_by_cluster, names_by_id, K,
            judge_model=args.judge_model,
            n_per_cluster=args.n_per_cluster,
            seed=args.seed,
        )

        result = {
            "K": K,
            "judge": args.judge_model,
            "kappa": kappa_result["kappa"],
            "accuracy": kappa_result["accuracy"],
            "n_sample": kappa_result["n_sample"],
            "per_cluster": kappa_result["per_cluster"],
            "cluster_names": {str(cid): v["name"] for cid, v in names_by_id.items()},
            "cluster_sizes": {str(cid): sz for cid, sz in cluster_sizes.items()},
        }

        out_path.write_text(json.dumps(result, indent=2))
        print(f"  κ = {result['kappa']:.3f}  acc = {result['accuracy']:.3f}", flush=True)
        print(f"  Saved: {out_path}", flush=True)
        summary[K] = result["kappa"]

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("K SWEEP SUMMARY")
    print("="*60)
    best_k = max((k for k in summary if summary[k] is not None),
                 key=lambda k: summary[k], default=None)
    for K_val in sorted(summary.keys()):
        marker = " ← best" if K_val == best_k else ""
        print(f"  K={K_val}: κ={summary[K_val]}{marker}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"k_sweep": summary, "judge": args.judge_model}, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
