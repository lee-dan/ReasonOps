#!/usr/bin/env python3
"""
Generate fig_stability.pdf: cosine similarity distributions for the operator
naming stability analysis.

Three distributions:
  - within-real:  pairwise cosines between descriptions of the same real
                  cluster across 30 exemplar seeds
  - across-real:  pairwise cosines between descriptions of different real
                  clusters
  - random-group: pairwise cosines between descriptions of 7 random span
                  groupings (no operator signal — null reference)

Usage:
    python -m reasonops.figures.gen_fig_stability \\
        --spans       data/operators/discovered_k7/spans_clustered.jsonl \\
        --names       data/operators/discovered_k7/cluster_names.json \\
        --out         figures \\
        --seeds       30 \\
        --n_exemplars 5

Requires: ANTHROPIC_API_KEY (cluster naming) and an embedding API key
          (OpenRouter or OpenAI for text-embedding-3-small).
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from anthropic import Anthropic

NAMING_MODEL = "claude-sonnet-4-6"
EMBED_MODEL  = "text-embedding-3-small"
N_ACROSS     = 2000
N_RANDOM     = 2000
RANDOM_FAKE_K = 7

NAMING_PROMPT = """\
Below are the top anchor phrases for a cluster of reasoning spans, followed by {n} example spans.

Top anchor phrases:
{anchors}

Example spans:
{examples}

In one sentence, name and describe the reasoning move this cluster represents.
Be concise and specific — capture what is cognitively distinctive about these spans."""


def load_spans(path):
    spans = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            spans[r["cluster"]].append(r)
    return spans


def load_names(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): (v["name"] if isinstance(v, dict) else v)
            for k, v in raw.items()}


def get_top_anchors(spans_by_cluster, cluster_id, topn=12):
    from collections import Counter
    ctr = Counter()
    for s in spans_by_cluster[cluster_id]:
        anchor = s.get("anchor", "")
        if anchor:
            ctr[anchor] += 1
    return [a for a, _ in ctr.most_common(topn)]


def call_naming(client_anthropic, anchors, examples):
    prompt = NAMING_PROMPT.format(
        n=len(examples),
        anchors="\n".join(f"  {a}" for a in anchors),
        examples="\n".join(f"  [{i+1}] {e[:200]}" for i, e in enumerate(examples)),
    )
    resp = client_anthropic.messages.create(
        model=NAMING_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def embed_texts(client_openai, texts, batch=512):
    embs = []
    for i in range(0, len(texts), batch):
        resp = client_openai.embeddings.create(
            model=EMBED_MODEL,
            input=texts[i:i+batch],
        )
        embs.extend([d.embedding for d in resp.data])
    mat = np.array(embs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat


def pairwise_cosine(mat_a, mat_b=None):
    if mat_b is None:
        return mat_a @ mat_a.T
    return mat_a @ mat_b.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans",      default="data/operators/discovered_k7/spans_clustered.jsonl")
    ap.add_argument("--names",      default="data/operators/discovered_k7/cluster_names.json")
    ap.add_argument("--out",        default="figures")
    ap.add_argument("--seeds",      type=int, default=30)
    ap.add_argument("--n_exemplars",type=int, default=5)
    ap.add_argument("--cache",      default="")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    spans_by_cluster = load_spans(args.spans)
    names = load_names(args.names)
    K = len(names)
    cluster_ids = sorted(names.keys())
    print(f"K={K}, clusters: {cluster_ids}")

    cache_path = Path(args.cache) if args.cache else Path(args.out) / "stability_cache.npz"

    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        real_embs  = cache["real_embs"]   # (K, seeds, dim)
        rand_embs  = cache["rand_embs"]   # (K, seeds, dim)
    else:
        # --- Real clusters: 30 seeds × K clusters ---
        print(f"Generating {args.seeds} seeds × {K} clusters = {args.seeds*K} naming calls...")
        all_anchors = {c: get_top_anchors(spans_by_cluster, c) for c in cluster_ids}
        rng = random.Random(42)

        real_descs = []  # list of (cluster_id, seed, description)
        for seed in range(args.seeds):
            for cid in cluster_ids:
                pool = spans_by_cluster[cid]
                sample = rng.sample(pool, min(args.n_exemplars, len(pool)))
                examples = [s.get("text", s.get("anchor", ""))[:300] for s in sample]
                desc = call_naming(anthropic_client, all_anchors[cid], examples)
                real_descs.append((cid, seed, desc))
                if len(real_descs) % 10 == 0:
                    print(f"  {len(real_descs)}/{args.seeds*K}", flush=True)

        # --- Random groups: 7 fake clusters from uniform span sample ---
        print("Generating random-group control...")
        all_spans = [s for spans in spans_by_cluster.values() for s in spans]
        rng2 = random.Random(99)
        rng2.shuffle(all_spans)
        fake_size = 2000
        rand_descs = []
        for fake_id in range(RANDOM_FAKE_K):
            fake_pool = all_spans[fake_id*fake_size:(fake_id+1)*fake_size]
            fake_anchors = [s.get("anchor","") for s in rng2.sample(fake_pool, 12)]
            for seed in range(args.seeds):
                sample = rng2.sample(fake_pool, args.n_exemplars)
                examples = [s.get("text", s.get("anchor",""))[:300] for s in sample]
                desc = call_naming(anthropic_client, fake_anchors, examples)
                rand_descs.append((fake_id, seed, desc))
        print(f"  {len(rand_descs)} random descriptions")

        # --- Embed everything ---
        print("Embedding descriptions...")
        real_texts = [d for _, _, d in real_descs]
        rand_texts = [d for _, _, d in rand_descs]
        real_mat = embed_texts(openai_client, real_texts)  # (K*seeds, dim)
        rand_mat = embed_texts(openai_client, rand_texts)  # (K*seeds, dim)

        # Reshape real: (K, seeds, dim)
        real_embs = real_mat.reshape(args.seeds, K, -1).transpose(1, 0, 2)  # (K, seeds, dim)
        rand_embs = rand_mat.reshape(args.seeds, RANDOM_FAKE_K, -1).transpose(1, 0, 2)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, real_embs=real_embs, rand_embs=rand_embs)
        print(f"Saved cache: {cache_path}")

    K_actual, S, dim = real_embs.shape
    print(f"real_embs: {real_embs.shape}, rand_embs: {rand_embs.shape}")

    # --- Compute distributions ---
    # Within-real: all C(seeds,2) pairs per cluster, pooled
    within_cosines = []
    for c in range(K_actual):
        mat = real_embs[c]  # (seeds, dim)
        sim = mat @ mat.T
        idx = np.triu_indices(S, k=1)
        within_cosines.extend(sim[idx].tolist())

    # Across-real: random pairs from different clusters
    rng3 = np.random.default_rng(7)
    all_real_flat = real_embs.reshape(-1, dim)  # (K*seeds, dim)
    cluster_labels = np.repeat(np.arange(K_actual), S)
    across_cosines = []
    while len(across_cosines) < N_ACROSS:
        i, j = rng3.integers(0, len(all_real_flat), size=2)
        if cluster_labels[i] != cluster_labels[j]:
            across_cosines.append(float(all_real_flat[i] @ all_real_flat[j]))
    across_cosines = across_cosines[:N_ACROSS]

    # Random-group control: within each fake cluster
    rand_within = []
    for c in range(rand_embs.shape[0]):
        mat = rand_embs[c]
        sim = mat @ mat.T
        idx = np.triu_indices(S, k=1)
        rand_within.extend(sim[idx].tolist())

    # Stats
    w  = np.array(within_cosines)
    ac = np.array(across_cosines)
    rw = np.array(rand_within)
    print(f"Within-real   n={len(w):,}  median={np.median(w):.3f}")
    print(f"Across-real   n={len(ac):,}  median={np.median(ac):.3f}")
    print(f"Random-group  n={len(rw):,}  median={np.median(rw):.3f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    from scipy.stats import gaussian_kde
    xs = np.linspace(-0.1, 1.05, 400)
    for data, label, color, ls in [
        (w,  f"Within-cluster (real operators)\nmedian = {np.median(w):.3f}", "#1e3a5c", "-"),
        (ac, f"Across-cluster baseline\nmedian = {np.median(ac):.3f}",        "#888888", "--"),
        (rw, f"Random-group control\nmedian = {np.median(rw):.3f}",           "#c0392b", ":"),
    ]:
        kde = gaussian_kde(data, bw_method=0.15)
        ax.plot(xs, kde(xs), lw=2.0, color=color, ls=ls, label=label)
        ax.axvline(np.median(data), color=color, lw=0.8, ls=ls, alpha=0.6)

    ax.set_xlabel("Pairwise cosine similarity of cluster descriptions", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("Operator naming stability (30 exemplar seeds)", fontsize=9, pad=6)
    plt.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_stability.{ext}", bbox_inches="tight", dpi=180)
        print(f"Saved: {out}/fig_stability.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
