#!/usr/bin/env python3
"""
Partial-trace LLM self-judge baseline for early correctness prediction.

Same as llm_judge.py but gives each model only the first p% of its own
reasoning trace (by character count), at depths 10/25/50/75/100%.
Produces a WP-AUC vs trace-depth curve comparable to the OST curve.

Usage:
    python -m reasonops.prediction.llm_judge_partial \
        --corpus  /scratch/.../final_dataset.jsonl.gz \
        --output  /scratch/.../predictions/preds_llm_judge_partial.jsonl \
        --datasets aime,gpqa,livecodebench \
        --workers 4

Requires: OPENROUTER_API_KEY in environment or .env.
"""
import argparse
import asyncio
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics import roc_auc_score

from reasonops.data.models import MODEL_REGISTRY
from reasonops.utils import EXCLUDE_DATASETS, FRACS

MAX_REASONING_CHARS = 12_000

FRACS = [10, 25, 50, 75, 100]

PROMPT_TEMPLATE = """\
You are evaluating an AI model's chain-of-thought reasoning on a {dataset} problem.

Question:
{question}

The model's reasoning so far (first {pct}% of its trace):
{reasoning}

Based only on the reasoning shown — without using outside knowledge about \
the correct answer — is the model on track to arrive at the CORRECT final answer?

Reply with ONLY one word: "correct" or "incorrect"."""

SKIP_MODELS = {"qwq-32b"}

OR_ID_FIXES = {
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-20250514":  "anthropic/claude-sonnet-4.5",
}


def get_or_model_id(model_key):
    if model_key in SKIP_MODELS:
        return None
    if model_key not in MODEL_REGISTRY:
        return OR_ID_FIXES.get(model_key, model_key)
    entry = MODEL_REGISTRY[model_key]
    mid = entry["or_model_id"]
    mid = OR_ID_FIXES.get(mid, mid)
    if entry.get("backend") == "anthropic" and "/" not in mid:
        return f"anthropic/{mid}"
    return mid


def truncate(text, pct):
    n = max(1, int(len(text) * pct / 100))
    return text[:n]


def parse_pred(text):
    if not text:
        return None
    t = text.strip().lower()
    if "incorrect" in t:
        return 0.0
    if "correct" in t:
        return 1.0
    return None


def wp_auc(rows_ds, key):
    groups = defaultdict(list)
    for r in rows_ds:
        v = r.get(key)
        if v is not None:
            groups[r["problem_id"]].append((r["correct"], v))
    aucs = [roc_auc_score([x[0] for x in g], [x[1] for x in g])
            for g in groups.values() if len(set(x[0] for x in g)) == 2]
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


async def judge_one(r, client, sem, frac):
    or_model_id = get_or_model_id(r.get("model", ""))
    if or_model_id is None:
        return None
    reasoning = (r.get("reasoning") or "").strip()
    question = (r.get("problem") or "").strip()
    if not reasoning:
        return None
    snippet = truncate(reasoning, frac)
    prompt = PROMPT_TEMPLATE.format(
        dataset=r.get("dataset", "unknown"),
        question=question[:1000],
        pct=frac,
        reasoning=snippet[:MAX_REASONING_CHARS],
    )
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=or_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0,
                )
                msg = resp.choices[0].message
                text = msg.content or ""
                if not text.strip():
                    dump = msg.model_dump()
                    rc = dump.get("reasoning") or dump.get("reasoning_content") or ""
                    text = rc if isinstance(rc, str) else ""
                return parse_pred(text)
            except Exception as e:
                if attempt < 5:
                    await asyncio.sleep(min(60, 4 ** attempt))
                else:
                    print(f"  FAILED {r['trace_id']} frac={frac}: {e}", flush=True)
    return None


async def run(args):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    opener = gzip.open if str(args.corpus).endswith(".gz") else open
    with opener(args.corpus, "rt") as f:
        all_records = [json.loads(line) for line in f]

    only_ds = set(d.strip() for d in args.datasets.split(",")) if args.datasets else None
    only_models = set(m.strip() for m in args.models.split(",")) if args.models else None
    records = [
        r for r in all_records
        if r.get("correct") is not None
        and r.get("reasoning")
        and (only_ds is None or r.get("dataset", "") in only_ds)
        and (only_models is None or r.get("model", "") in only_models)
    ]
    print(f"Loaded {len(records):,} traces  ds={only_ds}  models={only_models}", flush=True)

    # Load checkpoint
    out_path = Path(args.output)
    done = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if all(f"d{p}" in row for p in FRACS):
                        done[row["trace_id"]] = row
                except Exception:
                    pass
        print(f"  Resuming — {len(done):,} complete", flush=True)

    todo = [r for r in records if r["trace_id"] not in done]
    print(f"  {len(todo):,} remaining ({len(FRACS)} fracs each)", flush=True)

    if todo:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(args.workers)

        async def process_trace(r):
            results = await asyncio.gather(*[
                judge_one(r, client, sem, frac) for frac in FRACS
            ])
            return {
                "trace_id":   r["trace_id"],
                "dataset":    r.get("dataset", ""),
                "model":      r.get("model", ""),
                "problem_id": r.get("problem_id", ""),
                "correct":    int(bool(r.get("correct"))),
                **{f"d{p}": v for p, v in zip(FRACS, results)},
            }

        mode = "a" if out_path.exists() else "w"
        with open(out_path, mode) as fout:
            tasks = [process_trace(r) for r in todo]
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                row = await coro
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                done[row["trace_id"]] = row
                if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                    print(f"  {i+1:,}/{len(tasks):,}", flush=True)

    # Summary
    print("\n── WP-AUC by dataset and depth ──", flush=True)
    by_ds = defaultdict(list)
    for row in done.values():
        by_ds[row["dataset"]].append(row)

    for ds in sorted(by_ds):
        row_aucs = []
        for frac in FRACS:
            auc, _ = wp_auc(by_ds[ds], f"d{frac}")
            row_aucs.append(f"d{frac}={auc:.3f}")
        print(f"  {ds:<20} {' '.join(row_aucs)}", flush=True)

    print(f"\nSaved → {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",   required=True)
    ap.add_argument("--output",   required=True)
    ap.add_argument("--datasets", default="aime,gpqa,livecodebench")
    ap.add_argument("--models",   default="", help="comma-separated model names to process (empty = all)")
    ap.add_argument("--workers",  type=int, default=32)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
