#!/usr/bin/env python3
"""
Scope judge: classify Backtracking / Hypothesizing events as
LOCAL / SUB_PROBLEM / GLOBAL on GPQA reasoning traces.

Reproduces the supplemental table in `analyses/backtracking_local/`.

For each span of the target operator, a slice of the full reasoning trace
is marked from the event to the next span of the same operator (or trace
end), and Claude Sonnet 4.6 classifies the scope.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m reasonops.analysis.scope_judge \\
        --corpus data/final_dataset.jsonl.gz \\
        --operator BACKTRACKING --n 1500 --workers 12 \\
        --output results/scope/scope_BACKTRACKING.tsv
"""
import argparse
import gzip
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import pandas as pd

RUBRIC = """You are analyzing reasoning traces from large language models solving GPQA \
(graduate-level science) questions. Classify the SCOPE of a single \
revision-adjacent region marked in the trace --- a contiguous section where \
the model re-checks, considers an alternative, hypothesizes, qualifies, or \
hesitates. The marked region (between >>>MARKER<<< and <<<END>>>) starts at \
the target operator event and ends just before the next event of the SAME \
operator type (or at the end of the trace).

LOCAL: the model is re-checking a single calculation, fact lookup, or specific \
claim (e.g., "wait, 3*7=21 not 18"); the overall approach is unchanged.

SUB_PROBLEM: the model re-opens or alternates among specific cases, branches, \
or sub-questions (e.g., "let me reconsider case 2"); strategy preserved but a \
discrete piece is redone.

GLOBAL: the model abandons or proposes to replace its current solution \
strategy with a fundamentally different approach (e.g., "this Lagrangian \
approach isn't working, let me try energy conservation"); the high-level plan \
changes.

Output a single JSON object on one line:
  {"scope": "LOCAL"|"SUB_PROBLEM"|"GLOBAL", "rationale": "<one short sentence>"}"""


# ---------------------------------------------------------------------
# Event extraction from the corpus
# ---------------------------------------------------------------------

def load_events(corpus_path, operator):
    """Walk final_dataset.jsonl.gz and emit one event per matching span on GPQA."""
    events = []
    with gzip.open(corpus_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dataset") != "gpqa":
                continue
            spans = r.get("spans", [])
            for i, s in enumerate(spans):
                if s.get("operator") != operator:
                    continue
                events.append({
                    "trace_id": r["trace_id"],
                    "model":    r["model"],
                    "correct":  bool(r.get("correct")),
                    "span_idx": i,
                    "problem":  r.get("problem", ""),
                    "reasoning": r.get("reasoning", ""),
                    "spans":    spans,
                })
    return events


def stratified_sample(events, n_total, seed=0):
    rng = random.Random(seed)
    by_cell = {}
    for e in events:
        by_cell.setdefault((e["model"], e["correct"]), []).append(e)
    n_per_cell = max(1, n_total // max(1, len(by_cell)))
    out = []
    for grp in by_cell.values():
        out.extend(rng.sample(grp, min(n_per_cell, len(grp))))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------
# Marker prompt construction
# ---------------------------------------------------------------------

def build_prompt(event, operator):
    """Mark [target span, next same-operator span) inside the full reasoning."""
    reasoning = event["reasoning"]
    spans = event["spans"]
    target_idx = event["span_idx"]
    if not reasoning or not spans:
        return None

    # Locate each anchor sentence's char offset, walking forward.
    offsets = {}
    cursor = 0
    for i, s in enumerate(spans):
        anchor = (s.get("topic_sentence") or "").strip()
        if not anchor:
            continue
        idx = reasoning.find(anchor, cursor)
        if idx < 0:
            idx = reasoning.find(anchor[:60], cursor)
        if idx >= 0:
            offsets[i] = idx
            cursor = idx + len(anchor)

    if target_idx not in offsets:
        return None

    start = offsets[target_idx]
    later = [offsets[i] for i, s in enumerate(spans)
             if i > target_idx and s.get("operator") == operator and i in offsets]
    end = min(later) if later else len(reasoning)

    marked = (reasoning[:start] + ">>>MARKER<<<" + reasoning[start:end]
              + "<<<END>>>" + reasoning[end:])

    return (
        f"PROBLEM:\n{event['problem'].strip()}\n\n"
        f"FULL REASONING TRACE (the marked region between >>>MARKER<<< and "
        f"<<<END>>> spans from one {operator} event to the next):\n\n"
        f"{marked}\n\n"
        f"Classify the SCOPE of the marked region as LOCAL, SUB_PROBLEM, or "
        f"GLOBAL. Respond with one JSON object only."
    )


# ---------------------------------------------------------------------
# Anthropic call + parse
# ---------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def call_judge(prompt, judge_model, retries=4):
    client = _get_client()
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=judge_model,
                max_tokens=200,
                system=[{"type": "text", "text": RUBRIC,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
            else:
                raise


def parse_label(text):
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            scope = str(obj.get("scope", "")).upper().strip().replace("SUBPROBLEM", "SUB_PROBLEM")
            if scope in ("LOCAL", "SUB_PROBLEM", "GLOBAL"):
                return scope
        except Exception:
            pass
    for tok in ("SUB_PROBLEM", "LOCAL", "GLOBAL"):
        if re.search(r"\b" + tok + r"\b", text, re.IGNORECASE):
            return tok
    return "UNCLEAR"


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def _judge_one(event, operator, judge_model):
    prompt = build_prompt(event, operator)
    if prompt is None:
        return event, "MISSING"
    try:
        return event, parse_label(call_judge(prompt, judge_model))
    except Exception:
        return event, "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="Path to final_dataset.jsonl.gz")
    ap.add_argument("--operator", required=True,
                    choices=["BACKTRACKING", "HYPOTHESIZING"])
    ap.add_argument("--output", required=True, help="Output TSV path")
    ap.add_argument("--n_traces", type=int, default=1500,
                    help="Target sample size (stratified by model x correct)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge_model", default="claude-sonnet-4-6",
                    help="Anthropic model used for scope classification")
    args = ap.parse_args()

    print(f"Loading {args.operator} events from {args.corpus}...", flush=True)
    events = load_events(args.corpus, args.operator)
    print(f"  {len(events):,} events on GPQA", flush=True)
    sample = stratified_sample(events, args.n_traces, seed=args.seed)
    print(f"  stratified sample: {len(sample):,}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"LOCAL": 0, "SUB_PROBLEM": 0, "GLOBAL": 0}
    n_unclear = n_err = n_done = 0
    write_lock = threading.Lock()

    with open(out_path, "w") as fout:
        fout.write("trace_id\tspan_idx\toperator\tmodel\tcorrect\tscope\n")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_judge_one, e, args.operator, args.judge_model) for e in sample]
            for fu in as_completed(futures):
                event, scope = fu.result()
                if scope in counts:
                    counts[scope] += 1
                elif scope == "UNCLEAR":
                    n_unclear += 1
                else:
                    n_err += 1
                    continue
                with write_lock:
                    fout.write(f"{event['trace_id']}\t{event['span_idx']}\t"
                               f"{args.operator}\t{event['model']}\t"
                               f"{event['correct']}\t{scope}\n")
                    fout.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  [{n_done}/{len(sample)}] {counts} "
                          f"unclear={n_unclear} err={n_err}", flush=True)

    total = sum(counts.values())
    print(f"\nDone. n={total}", flush=True)
    if total:
        for k in ("LOCAL", "SUB_PROBLEM", "GLOBAL"):
            print(f"  {k}: {counts[k]/total*100:.1f}% ({counts[k]})", flush=True)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
