"""Cache sentence starts and pivot indices for sensitivity analysis."""

from reasonops.paths import DATA_DIR, OUTPUT_DIR, SPANS_FILE

HERE = OUTPUT_DIR / "sensitivity"
import argparse
import gzip
import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from reasonops.operators.discover_operators import (
    KEEP_MODELS,
    alpha_tokens,
    sentence_start,
    split_sentences,
)

CORPUS = DATA_DIR / "final_dataset.jsonl.gz"
SPANS = SPANS_FILE
NGRAM_LIST = DATA_DIR / "k_sweep/ngram_list.json"
CACHE = HERE / "cache"
TOP_VOCAB_MAX = 4000


def build_start():
    tok_freq = Counter()
    traces = {}
    n = 0
    with gzip.open(CORPUS, "rt") as f:
        for line in f:
            n += 1
            if n % 5000 == 0:
                print(f"  {n:,} traces...", flush=True)
            r = json.loads(line)
            model = r.get("model", "")
            if KEEP_MODELS and model not in KEEP_MODELS:
                continue
            starts = []
            for s in split_sentences(r.get("reasoning", "") or ""):
                tok_freq.update(alpha_tokens(s))
                st = sentence_start(s, n=5)
                if st:
                    starts.append(st)
            tid = (
                r.get("trace_id")
                or f"{model}__{r.get('dataset')}__{r.get('problem_id')}__{r.get('sample')}"
            )
            traces[tid] = {
                "model": model,
                "dataset": r.get("dataset", ""),
                "problem_id": r.get("problem_id", ""),
                "correct": None
                if r.get("correct") is None
                else str(r.get("correct")).lower() in ("true", "1"),
                "starts": starts,
            }
    out = CACHE / "start_cache.pkl"
    top_vocab = [t for t, _ in tok_freq.most_common(TOP_VOCAB_MAX)]
    with open(out, "wb") as fh:
        pickle.dump({"top_vocab_ranked": top_vocab, "traces": traces}, fh, protocol=4)
    print(
        f"wrote {out}  ({out.stat().st_size / 1000000.0:.1f} MB)  traces={len(traces):,}",
        flush=True,
    )


def build_trace():
    if NGRAM_LIST.exists():
        ng_list = json.loads(NGRAM_LIST.read_text())
    else:
        from reasonops.operators.pivot_dictionary import build_pivots

        ng_list = sorted(build_pivots(SPANS))
        NGRAM_LIST.parent.mkdir(parents=True, exist_ok=True)
        NGRAM_LIST.write_text(json.dumps(ng_list))
    ng2idx = {ng: i for i, ng in enumerate(ng_list)}
    print(f"pivot vocab: {len(ng_list):,}", flush=True)
    meta = {}
    rows = defaultdict(list)
    n_lines = n_oov = 0
    with open(SPANS) as f:
        for line in f:
            n_lines += 1
            if n_lines % 1000000 == 0:
                print(f"  {n_lines:,} spans...", flush=True)
            r = json.loads(line)
            ng = r.get("ngram") or ""
            if not ng:
                continue
            idx = ng2idx.get(ng)
            if idx is None:
                n_oov += 1
                continue
            tid = r["trace_id"]
            if tid not in meta:
                meta[tid] = (
                    r.get("model", ""),
                    r.get("dataset", ""),
                    r.get("problem_id", ""),
                    r.get("correct"),
                )
            rows[tid].append((r.get("span_idx", 0), idx, r.get("cluster", -1)))
    print(
        f"streamed {n_lines:,} spans | OOV skipped {n_oov:,} | traces with >=1 pivot: {len(rows):,}",
        flush=True,
    )
    traces = {}
    for tid, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        m = meta[tid]
        traces[tid] = {
            "model": m[0],
            "dataset": m[1],
            "problem_id": m[2],
            "correct": m[3],
            "pivot_idx": np.array([x[1] for x in lst], dtype=np.int32),
            "default_cluster": np.array([x[2] for x in lst], dtype=np.int8),
        }
    out = CACHE / "trace_cache.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"ngram_list": ng_list, "traces": traces}, fh, protocol=4)
    print(f"wrote {out}  ({out.stat().st_size / 1000000.0:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["start", "trace", "all"], default="all")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.which in ("start", "all"):
        build_start()
    if args.which in ("trace", "all"):
        build_trace()


if __name__ == "__main__":
    main()
