#!/usr/bin/env python3
"""
Classification eval: given the K-means cluster names, sample N held-out spans
per cluster and ask an independent LLM judge to classify each into one of the
named operator categories. Compute Cohen's kappa between the LLM's labels and
the K-means cluster assignments.

Usage:
    python -m reasonops.eval.judge_validation \\
        --names         .../cluster_names.json \\
        --spans         .../spans_clustered.jsonl \\
        --viz           .../umap_viz.jsonl \\
        --output        .../classification_kappa.json \\
        --judge         openai/gpt-5-mini \\
        --n_per_cluster 50
"""
import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import aiohttp
import anthropic as _anthropic_module
import openai as _openai_module
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix


def topic_sentence(span: dict, max_len: int = 120, full_span: bool = False) -> str:
    text = span.get("anchor", "")
    body = span.get("body", [])
    if full_span:
        if body:
            text += " " + " ".join(body)
        return text[:max_len].strip()
    # Default: first sentence only
    if body:
        text += " " + " ".join(body)
    m = re.search(r"[.!?]\s", text)
    if m:
        text = text[:m.start() + 1]
    return text[:max_len].strip()


def build_classify_prompt(names: dict, topic: str, prior: str = None,
                          full_span: bool = False, exemplars: dict = None) -> str:
    cats_lines = []
    for cid, n in sorted(names.items(), key=lambda x: int(x[0])):
        line = f"  {int(cid)+1}. {n['name']}: {n['description']}"
        phrases = n.get("top_starts", [])
        if phrases:
            ph_str = ", ".join(f'"{p}"' for p in phrases[:12])
            line += f"\n     Typical openers: {ph_str}"
        if exemplars and int(cid) in exemplars:
            for i, ex in enumerate(exemplars[int(cid)][:3], 1):
                line += f'\n     Example {i}: "{ex}"'
        cats_lines.append(line)
    cats = "\n".join(cats_lines)
    context_block = ""
    if prior:
        context_block = f"\nPreceding sentence (for context):\n  \"{prior}\"\n"
    span_label = "Target span" if full_span else "Target sentence"
    return f"""You are classifying a sentence from an LLM reasoning trace.

Each category below has a description AND the typical opening phrases that define it.
Classify the {span_label} into the two most likely categories based on which opening phrases it matches AND the cognitive move being made:

{cats}
{context_block}
{span_label}:
  "{topic}"

Respond with ONLY two category numbers separated by a comma, best match first (e.g. "3,7"). No explanation."""


def _parse_nums(text: str):
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    top1 = int(nums[0]) - 1
    top2 = int(nums[1]) - 1 if len(nums) > 1 else None
    return (top1, top2)


def _classify_openai_sync(model: str, prompt: str) -> tuple | None:
    import time
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = _openai_module.OpenAI(api_key=api_key)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_nums(text)
        except Exception as e:
            print(f"  OpenAI error: {e}", flush=True)
            time.sleep(2 ** (attempt + 1))
    return None


def _classify_anthropic_sync(model: str, prompt: str) -> tuple | None:
    import time
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = _anthropic_module.Anthropic(api_key=api_key)
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=64,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            return _parse_nums(text)
        except Exception as e:
            print(f"  Anthropic error: {e}", flush=True)
            time.sleep(2 ** (attempt + 1))
    return None


async def classify(session, judge, prompt, api_key,
                   max_tokens: int = 2000, temperature: float = 0.0):
    is_anthropic = judge.startswith("anthropic/") or judge.startswith("claude-")
    if is_anthropic:
        model = judge.replace("anthropic/", "")
        return await asyncio.to_thread(_classify_anthropic_sync, model, prompt)

    is_openai = judge.startswith("openai/") or judge.startswith("gpt-") or judge.startswith("o1") or judge.startswith("o3")
    if is_openai:
        model = judge.replace("openai/", "")
        return await asyncio.to_thread(_classify_openai_sync, model, prompt)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # High max_tokens because reasoning models consume tokens on hidden thinking.
    payload = {"model": judge, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    for attempt in range(5):
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                     json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                if resp.status != 200:
                    body = await resp.text()
                    print(f"status={resp.status}: {body[:200]}", flush=True)
                    await asyncio.sleep(3)
                    continue
                data = await resp.json()
                msg = data["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning") or "").strip()
                return _parse_nums(text)
        except Exception as e:
            print(f"exc: {e}", flush=True)
            await asyncio.sleep(3)
    return None


def empirical_random_kappa(y_true, classes, n_trials=1000, seed=42):
    """Empirical random baseline: assign uniform random labels, compute κ.

    Returns (mean_kappa, std_kappa) across n_trials. Expected value ≈ 0 by
    construction of Cohen's κ; std shows finite-sample variance.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    K = len(classes)
    n = len(y_true)
    kappas = []
    for _ in range(n_trials):
        y_rand = rng.randint(0, K, size=n).tolist()
        try:
            kappas.append(cohen_kappa_score(y_true, y_rand, labels=list(classes)))
        except Exception:
            pass
    return float(np.mean(kappas)), float(np.std(kappas))


async def run(args, names, api_key):
    # Sample N spans per K-means cluster (balanced)
    rng = random.Random(args.seed)
    cluster_ids = sorted(int(k) for k in names.keys())

    # Index all spans by (trace_id, span_idx) for prior-context lookup
    by_cluster = defaultdict(list)
    span_index = {}
    with open(args.spans) as f:
        for line in f:
            r = json.loads(line)
            c = r.get("cluster", -2)
            tid = r.get("trace_id", "")
            idx = r.get("span_idx", 0)
            span_index[(tid, idx)] = r
            if c in cluster_ids and r.get("anchor", "").strip():
                by_cluster[c].append(r)

    print(f"Loaded spans: {sum(len(v) for v in by_cluster.values()):,} "
          f"across {len(by_cluster)} clusters")

    test_spans = []
    for c in cluster_ids:
        pool = by_cluster[c]
        k = min(args.n_per_cluster, len(pool))
        sampled = rng.sample(pool, k)
        test_spans.extend([(s, c) for s in sampled])

    print(f"Classifying {len(test_spans)} spans with {args.judge}...")

    # Collect exemplars from spans NOT in the test set
    exemplars: dict = {}
    if getattr(args, "n_exemplars", 0) > 0:
        test_ids = {id(s) for s, _ in test_spans}
        for c in cluster_ids:
            pool = [s for s in by_cluster[c] if id(s) not in test_ids]
            sample = rng.sample(pool, min(args.n_exemplars, len(pool)))
            exemplars[c] = [topic_sentence(s, max_len=200, full_span=True) for s in sample]
        print(f"  Injecting {args.n_exemplars} exemplars/category into prompt")

    semaphore = asyncio.Semaphore(args.workers)
    async with aiohttp.ClientSession() as session:
        async def one(span, true_c):
            async with semaphore:
                topic = topic_sentence(span, max_len=800, full_span=args.full_span)
                tid = span.get("trace_id", "")
                idx = span.get("span_idx", 0)
                prior_span = span_index.get((tid, idx - 1))
                prior = topic_sentence(prior_span, max_len=150) if prior_span else None
                prompt = build_classify_prompt(names, topic, prior=prior,
                                               full_span=args.full_span,
                                               exemplars=exemplars or None)
                result = await classify(session, args.judge, prompt, api_key,
                                        max_tokens=args.max_tokens,
                                        temperature=args.temperature)
                if result is None:
                    return (true_c, None, None, topic)
                top1, top2 = result
                return (true_c, top1, top2, topic)

        results = await asyncio.gather(*[one(s, c) for s, c in test_spans])

    valid = [(t, p1, p2, tpc) for t, p1, p2, tpc in results if p1 is not None]
    n_failed = len(results) - len(valid)
    print(f"  {n_failed} API failures" if n_failed else f"  all classified")

    y_true = [t for t, p1, p2, _ in valid]
    y_pred = [p1 for t, p1, p2, _ in valid]
    y_pred2 = [p2 for t, p1, p2, _ in valid]

    kappa = cohen_kappa_score(y_true, y_pred)
    acc1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    acc2 = sum(1 for t, p1, p2 in zip(y_true, y_pred, y_pred2)
               if t == p1 or t == p2) / len(y_true)

    # Empirical random baseline: uniformly random labels, 1000 trials
    random_kappa_mean, random_kappa_std = empirical_random_kappa(
        y_true, cluster_ids, n_trials=1000, seed=args.seed
    )

    print(f"\n=== Results ===")
    print(f"  N={len(y_true)} spans classified")
    print(f"  Accuracy@1: {acc1:.4f}")
    print(f"  Accuracy@2: {acc2:.4f}")
    print(f"  Cohen's κ:  {kappa:.4f}")
    print(f"  Random baseline κ (n=1000 trials): {random_kappa_mean:.4f} ± {random_kappa_std:.4f}")

    # Per-cluster F1
    cluster_names = {int(k): v["name"] for k, v in names.items()}
    target_names = [cluster_names[c] for c in cluster_ids]
    report_dict = classification_report(
        y_true, y_pred, labels=cluster_ids, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    print(f"\nPer-cluster F1:")
    for n in target_names:
        f1 = report_dict[n]["f1-score"]
        s = report_dict[n]["support"]
        print(f"  {n:<20} F1={f1:.3f} support={int(s)}")

    cm = confusion_matrix(y_true, y_pred, labels=cluster_ids)
    print(f"\nConfusion matrix (K-means rows × LLM-classifier cols):")
    print("         " + "  ".join(f"{n[:8]:>8}" for n in target_names))
    for i, name in enumerate(target_names):
        row = "  ".join(f"{c:>8}" for c in cm[i])
        print(f"  {name[:7]:<7}  {row}")

    out = {
        "judge": args.judge, "n_spans": len(y_true),
        "accuracy": float(acc1), "accuracy_at_2": float(acc2),
        "cohen_kappa": float(kappa),
        "random_kappa_mean": float(random_kappa_mean),
        "random_kappa_std": float(random_kappa_std),
        "per_cluster_f1": {n: report_dict[n]["f1-score"] for n in target_names},
        "confusion_matrix": cm.tolist(),
        "cluster_names": cluster_names,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--clusters", default=None,
                    help="clusters.json with op_stats.top_starts to inject into prompt")
    ap.add_argument("--full_span", action="store_true",
                    help="Show full span (anchor+body) instead of first sentence only")
    ap.add_argument("--n_exemplars", type=int, default=0,
                    help="Inject this many example spans per category into the prompt (0=disabled)")
    ap.add_argument("--judge", default="openai/gpt-5-mini")
    ap.add_argument("--max_tokens",   type=int,   default=2000)
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--n_per_cluster", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.names) as f:
        names = json.load(f)

    if args.clusters:
        with open(args.clusters) as f:
            clusters_data = json.load(f)
        op_stats = clusters_data.get("op_stats", {})
        for cid_str, stat in op_stats.items():
            if cid_str in names:
                names[cid_str]["top_starts"] = stat.get("top_starts", [])

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    is_anthropic_judge = args.judge.startswith("anthropic/") or args.judge.startswith("claude-")
    is_openai_judge = args.judge.startswith("openai/") or args.judge.startswith("gpt-") or args.judge.startswith("o1") or args.judge.startswith("o3")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not is_anthropic_judge and not is_openai_judge:
        raise ValueError("Set OPENROUTER_API_KEY")

    asyncio.run(run(args, names, api_key))


if __name__ == "__main__":
    main()
