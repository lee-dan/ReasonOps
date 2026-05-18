#!/usr/bin/env python3
"""
Measure end-to-end pipeline timing for the operator discovery pipeline.

Runs every stage on the full corpus and records wall-clock seconds:
  - Sentence segmentation + pivot extraction
  - e5-small-v2 embedding of accepted pivots
  - K-means clustering
  - Per-trace annotation latency (dict lookup)
  - Full-corpus annotation

Outputs: data/timing_benchmark.json

Usage:
    python -m reasonops.eval.benchmark_timing \\
        --corpus data/final_dataset.jsonl.gz \\
        --output data/timing_benchmark.json
"""
import argparse
import gzip, json, os, time
from pathlib import Path

import numpy as np

from reasonops.utils import EXCLUDE_DATASETS, N_OPERATORS

# Import the actual production functions
from reasonops.operators.discover_operators import (
    split_sentences, sentence_start, build_top_vocab, find_type_a_starts,
    segment_traces, load_e5, embed_pivots, cluster_type_a_starts,
    assign_span_clusters,
)

SEED = 42
TOP_VOCAB  = 2000
MIN_FREQ   = 100
MIN_DOMAIN = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/final_dataset.jsonl.gz")
    ap.add_argument("--output", default="data/timing_benchmark.json")
    ap.add_argument("--n_reps", type=int, default=1000)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    corpus_path = args.corpus

    results = {}

    # ── Load traces ──────────────────────────────────────────────────────────
    print("Loading traces...", flush=True)
    t0 = time.perf_counter()
    traces = []
    with gzip.open(corpus_path, "rt") as f:
        for line in f:
            traces.append(json.loads(line))
    load_time = time.perf_counter() - t0
    print(f"  {len(traces):,} traces in {load_time:.1f}s", flush=True)
    results["n_traces"] = len(traces)

    # ── Step 1: Sentence segmentation ────────────────────────────────────────
    print("\nStep 1: Sentence segmentation...", flush=True)
    t0 = time.perf_counter()
    trace_sentences = {}
    trace_meta      = {}
    for r in traces:
        tid  = r["trace_id"]
        text = r.get("thinking") or r.get("trace") or r.get("reasoning") or ""
        trace_sentences[tid] = split_sentences(text) if text else []
        trace_meta[tid]      = {"dataset": r.get("dataset", ""), "model": r.get("model", "")}
    seg_time = time.perf_counter() - t0
    print(f"  Segmented in {seg_time:.2f}s", flush=True)

    # ── Step 2: Pivot extraction ──────────────────────────────────────────────
    print("\nStep 2: Pivot extraction (vocab + type-A filter)...", flush=True)
    t0 = time.perf_counter()
    top_vocab = build_top_vocab(trace_sentences, top_n=TOP_VOCAB)
    type_a    = find_type_a_starts(trace_sentences, MIN_FREQ, top_vocab,
                                   trace_meta=trace_meta,
                                   min_domain_count=MIN_DOMAIN)
    pivot_time  = time.perf_counter() - t0
    pivot_total = seg_time + pivot_time
    print(f"  {len(type_a):,} pivots in {pivot_time:.2f}s (+seg = {pivot_total:.2f}s)", flush=True)
    results["n_pivots"]               = len(type_a)
    results["pivot_extraction_sec"]   = round(pivot_total, 2)
    results["pivot_extraction_claim"] = "<2 min (120 sec)"
    results["pivot_extraction_pass"]  = pivot_total < 120

    # ── Step 3: Embedding ─────────────────────────────────────────────────────
    print("\nStep 3: Embedding pivots with e5-small-v2...", flush=True)
    e5_model = load_e5()
    pivots_list = sorted(type_a, key=lambda x: " ".join(x))
    t0 = time.perf_counter()
    embeddings = embed_pivots(pivots_list, e5_model)
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(pivots_list)} pivots in {embed_time:.2f}s", flush=True)
    results["embed_sec"]   = round(embed_time, 2)
    results["embed_claim"] = "4 sec"
    results["embed_pass"]  = embed_time < 10  # 2.5x margin

    # ── Step 4: K-means ───────────────────────────────────────────────────────
    print("\nStep 4: K-means...", flush=True)
    t0 = time.perf_counter()
    start_to_cluster = cluster_type_a_starts(type_a, e5_model, n_operators=N_OPERATORS, seed=SEED)
    kmeans_time = time.perf_counter() - t0
    print(f"  KMeans in {kmeans_time:.3f}s", flush=True)
    results["kmeans_sec"]   = round(kmeans_time, 3)
    results["kmeans_claim"] = "instantaneous"
    results["kmeans_pass"]  = kmeans_time < 5

    # ── Step 5: End-to-end ───────────────────────────────────────────────────
    e2e = pivot_total + embed_time + kmeans_time
    results["e2e_sec"]   = round(e2e, 2)
    results["e2e_claim"] = "<5 min (300 sec)"
    results["e2e_pass"]  = e2e < 300
    print(f"\nEnd-to-end (seg+pivot+embed+kmeans): {e2e:.1f}s", flush=True)

    # ── Step 6: Segment full corpus + assign clusters ─────────────────────────
    print("\nStep 6: Segment + annotate full corpus...", flush=True)
    t0 = time.perf_counter()
    trace_spans = segment_traces(trace_sentences, frozenset(type_a))
    assign_span_clusters(trace_spans, start_to_cluster)
    corpus_time = time.perf_counter() - t0
    print(f"  Full corpus annotation: {corpus_time:.2f}s", flush=True)
    results["corpus_annotation_sec"]   = round(corpus_time, 2)
    results["corpus_annotation_claim"] = "<30 sec"
    results["corpus_annotation_pass"]  = corpus_time < 30

    # ── Step 7: Per-trace annotation latency ─────────────────────────────────
    print("\nStep 7: Per-trace annotation latency...", flush=True)
    sample_tid   = traces[0]["trace_id"]
    sample_sents = trace_sentences[sample_tid]
    t0 = time.perf_counter()
    for _ in range(args.n_reps):
        for s in sample_sents:
            start_to_cluster.get(sentence_start(s), -1)
    per_trace_ms = (time.perf_counter() - t0) / args.n_reps * 1000
    print(f"  {per_trace_ms:.3f} ms per trace (avg over {args.n_reps} reps)", flush=True)
    results["annotate_per_trace_ms"] = round(per_trace_ms, 3)
    results["annotate_claim"]        = "<1 ms"
    results["annotate_pass"]         = per_trace_ms < 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("TIMING BENCHMARK RESULTS", flush=True)
    print("="*60, flush=True)
    checks = [
        ("Pivot extraction",      "pivot_extraction_sec",   "pivot_extraction_claim", "pivot_extraction_pass", "sec"),
        ("Embedding pivots",      "embed_sec",              "embed_claim",            "embed_pass",            "sec"),
        ("K-means",               "kmeans_sec",             "kmeans_claim",           "kmeans_pass",           "sec"),
        ("Per-trace annotation",  "annotate_per_trace_ms",  "annotate_claim",         "annotate_pass",         "ms"),
        ("End-to-end discovery",  "e2e_sec",                "e2e_claim",              "e2e_pass",              "sec"),
        ("Full corpus annotation","corpus_annotation_sec",  "corpus_annotation_claim","corpus_annotation_pass","sec"),
    ]
    all_pass = True
    for label, mkey, ckey, pkey, unit in checks:
        measured = results[mkey]
        passed   = results[pkey]
        mark     = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {label:<26} {measured} {unit}  (claim: {results[ckey]})", flush=True)

    results["all_pass"] = all_pass
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
