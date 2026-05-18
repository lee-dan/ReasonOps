#!/usr/bin/env python3
"""
LLM self-judge correctness baseline.

For each trace, prompts the SAME OpenRouter model that generated the trace,
giving it the original question + full chain-of-thought + final response,
and asking whether the answer is correct.

No training — single score column per trace: llm_judge_prob (0.0 or 1.0).
Checkpointed: safe to resume if interrupted.

Usage:
    python -m reasonops.prediction.llm_judge \
        --corpus  data/final_dataset.jsonl.gz \
        --output  data/predictions/preds_llm_judge.jsonl.gz \
        [--workers 16]

Requires: OPENROUTER_API_KEY in environment (or .env file).
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
from reasonops.utils import EXCLUDE_DATASETS

MAX_REASONING_CHARS = 12_000
MAX_RESPONSE_CHARS = 2_000
MAX_RETRY_WAIT = 60

PROMPT_TEMPLATE = """\
You are evaluating an AI model's reasoning trace on a {dataset} problem.

Question:
{question}

The model's chain-of-thought reasoning:
{reasoning}

The model's final answer:
{response}

Based only on the reasoning and answer above — without using outside knowledge \
about the correct answer — did the model arrive at the CORRECT final answer?

Reply with ONLY one word: "correct" or "incorrect"."""


SKIP_MODELS = {"qwq-32b"}  # no longer available on OpenRouter

OR_ID_FIXES = {
    "claude-haiku-4-5-20251001":  "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-20250514":   "anthropic/claude-sonnet-4.5",
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


def build_prompt(r):
    question  = (r.get("problem") or "").strip()
    reasoning = (r.get("reasoning") or "").strip()
    response  = (r.get("model_response") or "").strip()
    if not reasoning:
        reasoning = response
    return PROMPT_TEMPLATE.format(
        dataset=r.get("dataset", "unknown"),
        question=question,
        reasoning=reasoning[:MAX_REASONING_CHARS],
        response=response[:MAX_RESPONSE_CHARS],
    )


def parse_pred(text):
    if not text:
        return None, None
    t = text.strip().lower()
    if "incorrect" in t:
        return 0, 0.0
    if "correct" in t:
        return 1, 1.0
    return None, None


def wp_auc(y_true, y_pred, pids):
    groups = defaultdict(list)
    for pid, yt, yp in zip(pids, y_true, y_pred):
        groups[pid].append((yt, yp))
    aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
            for v in groups.values() if len(set(p[0] for p in v)) == 2]
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


async def judge_one(r, client, sem):
    or_model_id = get_or_model_id(r.get("model", ""))
    if or_model_id is None:
        return None
    prompt = build_prompt(r)
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=or_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0,
                )
                msg  = resp.choices[0].message
                text = msg.content or ""
                # Reasoning models (r1-distill, kimi, nemotron) return content=None
                # with the answer in msg.reasoning (stored as a Pydantic extra field).
                # getattr misses extra fields — use model_dump() to access them.
                if not text.strip():
                    dump = msg.model_dump()
                    rc   = dump.get("reasoning") or dump.get("reasoning_content") or ""
                    text = rc if isinstance(rc, str) else ""
                pred, prob = parse_pred(text)
                return {
                    "trace_id":       r["trace_id"],
                    "dataset":        r.get("dataset", ""),
                    "model":          r.get("model", ""),
                    "problem_id":     r.get("problem_id", ""),
                    "correct":        int(bool(r.get("correct"))),
                    "raw":            text.strip(),
                    "llm_judge_pred": pred,
                    "llm_judge_prob": prob,
                }
            except Exception as e:
                if attempt < 5:
                    await asyncio.sleep(min(MAX_RETRY_WAIT, 4 ** attempt))
                else:
                    print(f"  FAILED {r['trace_id']} ({or_model_id}): {e}", flush=True)
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

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"Loading corpus: {args.corpus}", flush=True)
    opener_r = gzip.open if str(args.corpus).endswith(".gz") else open
    with opener_r(args.corpus, "rt") as f:
        all_records = [json.loads(line) for line in f]

    records = [
        r for r in all_records
        if r.get("dataset", "") not in EXCLUDE_DATASETS
        and r.get("correct") is not None
    ]
    if args.only_datasets:
        only = set(d.strip() for d in args.only_datasets.split(","))
        records = [r for r in records if r.get("dataset", "") in only]
        print(f"  Filtering to datasets: {sorted(only)}", flush=True)
    print(f"  {len(records):,} traces to evaluate", flush=True)

    # Load checkpoint
    done = {}
    out_path = Path(args.output)
    if out_path.exists():
        opener_c = gzip.open if str(args.output).endswith(".gz") else open
        with opener_c(args.output, "rt") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("llm_judge_pred") is not None:
                        done[row["trace_id"]] = row
                except Exception:
                    pass
        print(f"  Resuming — {len(done):,} already done", flush=True)

    todo = [r for r in records if r["trace_id"] not in done]
    print(f"  {len(todo):,} remaining", flush=True)

    if todo:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        opener_w = gzip.open if str(args.output).endswith(".gz") else open
        mode = "at" if out_path.exists() else "wt"
        sem = asyncio.Semaphore(args.workers)
        done_count = len(done)
        total = len(records)

        with opener_w(args.output, mode) as fout:
            tasks = [judge_one(r, client, sem) for r in todo]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    done[result["trace_id"]] = result
                done_count += 1
                n_valid = sum(1 for v in done.values()
                              if v.get("llm_judge_pred") is not None)
                if done_count % 100 == 0 or done_count == total:
                    print(f"  {done_count:,}/{total:,} ({done_count/total*100:.0f}%)"
                          f"  labeled={n_valid:,}", flush=True)

    # Summary
    print("\n── Results ──", flush=True)
    all_y, all_p, all_pid = [], [], []
    by_ds = defaultdict(lambda: {"y": [], "p": [], "pid": []})
    for r in records:
        row = done.get(r["trace_id"])
        if row is None or row.get("llm_judge_prob") is None:
            continue
        ds = row["dataset"]
        yt = int(bool(r["correct"]))
        pb = float(row["llm_judge_prob"])
        pid = r.get("problem_id", "")
        by_ds[ds]["y"].append(yt)
        by_ds[ds]["p"].append(pb)
        by_ds[ds]["pid"].append(pid)
        all_y.append(yt); all_p.append(pb); all_pid.append(pid)

    for ds in sorted(by_ds):
        d = by_ds[ds]
        if len(set(d["y"])) < 2:
            continue
        wauc, np_ = wp_auc(d["y"], d["p"], d["pid"])
        acc = float(np.mean(np.array(d["p"]).round() == np.array(d["y"])))
        print(f"  {ds:<22} WP-AUC={wauc:.4f}  acc={acc:.3f}"
              f"  n={len(d['y']):,}  problems={np_}", flush=True)
    if all_y:
        wauc, np_ = wp_auc(all_y, all_p, all_pid)
        acc = float(np.mean(np.array(all_p).round() == np.array(all_y)))
        print(f"  {'OVERALL':<22} WP-AUC={wauc:.4f}  acc={acc:.3f}"
              f"  n={len(all_y):,}  problems={np_}", flush=True)

    print(f"\nSaved → {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",        required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--workers",       type=int, default=16)
    ap.add_argument("--only_datasets", default="", help="comma-separated dataset names to process")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
