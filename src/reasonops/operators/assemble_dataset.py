#!/usr/bin/env python3
"""
Assemble the final dataset by joining raw traces with discovered operator spans.

Reads:
  - traces/sample_{1..N}/{model}/{dataset}/*.json   — raw inference output
  - spans_clustered.jsonl                           — output of discover_operators.py
  - cluster_names.json                              — output of discover_operators.py

Writes:
  - final_dataset.jsonl.gz   — one record per trace with operator_sequence, spans, correct

Usage:
    python -m reasonops.operators.assemble_dataset \\
        --traces /scratch/.../traces \\
        --spans  /scratch/.../operators/discovered_k7/spans_clustered.jsonl \\
        --names  /scratch/.../operators/discovered_k7/cluster_names.json \\
        --output /scratch/.../final_dataset.jsonl.gz
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from reasonops.utils import FAMILY


def build_spans(segments, operator_names):
    def full_sent(seg):
        text = seg.get("anchor", "")
        body = seg.get("body", [])
        return (text + " " + " ".join(body)).strip() if body else text.strip()

    preamble, spans, cur = [], [], None
    for s in segments:
        cid  = s["cluster"]
        sent = full_sent(s)
        if cid >= 0:
            if cur is not None:
                spans.append(cur)
            cur = {
                "operator":       operator_names[cid],
                "cluster":        int(cid),
                "topic_sentence": sent,
                "continuation":   [],
            }
        else:
            if cur is None:
                preamble.append(sent)
            else:
                cur["continuation"].append(sent)
    if cur is not None:
        spans.append(cur)
    return preamble, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True,
                    help="Root of traces dir: sample_{1..N}/{model}/{dataset}/*.json")
    ap.add_argument("--spans",  required=True,
                    help="spans_clustered.jsonl from discover_operators.py")
    ap.add_argument("--names",  required=True,
                    help="cluster_names.json from discover_operators.py")
    ap.add_argument("--output", required=True,
                    help="Output path (will gzip if .gz)")
    args = ap.parse_args()

    with open(args.names) as f:
        names = json.load(f)
    operator_names = {int(k): v["name"] for k, v in names.items()}

    print("Loading spans_clustered.jsonl...", flush=True)
    segs_by_trace = defaultdict(list)
    with open(args.spans) as f:
        for line in f:
            r = json.loads(line)
            segs_by_trace[r["trace_id"]].append({
                "span_idx": r.get("span_idx", 0),
                "anchor":   r.get("anchor", ""),
                "body":     r.get("body", []),
                "cluster":  r["cluster"],
            })
    for tid in segs_by_trace:
        segs_by_trace[tid].sort(key=lambda s: s["span_idx"])
    print(f"  {len(segs_by_trace):,} traces with span data", flush=True)

    root     = Path(args.traces)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"Walking {root}...", flush=True)
    with gzip.open(out_path, "wt", compresslevel=6, encoding="utf-8") as out:
        for sample_dir in sorted(root.glob("sample_*")):
            try:
                sample_idx = int(sample_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for model_dir in sorted(sample_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model  = model_dir.name
                family = FAMILY.get(model, "unknown")
                for ds_dir in sorted(model_dir.iterdir()):
                    if not ds_dir.is_dir():
                        continue
                    ds = ds_dir.name
                    for fp in ds_dir.glob("*.json"):
                        if fp.name.startswith("_"):
                            continue
                        try:
                            d = json.loads(fp.read_text())
                        except Exception:
                            continue

                        problem_id = d.get("problem_id", fp.stem)
                        tid        = f"{model}__{ds}__{problem_id}__s{sample_idx}"
                        segments   = segs_by_trace.get(tid, [])
                        preamble, spans = build_spans(segments, operator_names)

                        record = {
                            "trace_id":               tid,
                            "model":                  model,
                            "model_family":           family,
                            "dataset":                ds,
                            "problem_id":             problem_id,
                            "sample":                 sample_idx,
                            "problem":                d.get("problem", ""),
                            "gold_answer":            d.get("answer", ""),
                            "model_response":         d.get("response", ""),
                            "reasoning":              d.get("reasoning", ""),
                            "correct":                d.get("correct"),
                            "truncated":              bool(d.get("truncated", False)),
                            "finish_reason":          d.get("finish_reason", ""),
                            "n_sentences":            len(segments),
                            "n_spans":                len(spans),
                            "operator_sequence":      [s["cluster"]  for s in spans],
                            "operator_sequence_named":[s["operator"] for s in spans],
                            "preamble":               preamble,
                            "spans":                  spans,
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total += 1
                        if total % 5000 == 0:
                            print(f"  wrote {total:,} traces...", flush=True)

    print(f"\nDone: {total:,} traces → {out_path}", flush=True)
    print(f"Size: {out_path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
