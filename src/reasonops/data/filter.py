#!/usr/bin/env python3
"""
Filter the graded dataset to the clean training corpus.

Drops:
  - Models with incomplete runs (read from configs/data.toml dropped_models)
  - Excluded datasets (read from configs/data.toml excluded_datasets)
  - Truncated traces
  - Unlabeled traces (correct is None)

Usage:
    python -m reasonops.data.filter \\
        --input   /scratch/.../traces_graded.jsonl.gz \\
        --output  /scratch/.../final_dataset.jsonl.gz \\
        --summary /scratch/.../dataset_summary.json
"""
import argparse
import gzip
import json
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # for Python < 3.11
from collections import Counter, defaultdict
from pathlib import Path


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "data.toml"
    if not cfg_path.exists():
        sys.exit(f"configs/data.toml not found at {cfg_path}")
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    cfg            = load_config()
    dropped_models  = set(cfg.get("dropped_models", []))
    dropped_datasets = set(cfg.get("excluded_datasets", []))

    print(f"Dropping models:   {sorted(dropped_models)}")
    print(f"Dropping datasets: {sorted(dropped_datasets)}")

    opener = gzip.open if args.input.endswith(".gz") else open
    traces = []
    with opener(args.input, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    print(f"Loaded {len(traces):,} traces")

    skip_reasons = Counter()
    kept = []
    for t in traces:
        if t.get("model") in dropped_models:
            skip_reasons["dropped_model"] += 1
            continue
        if t.get("dataset") in dropped_datasets:
            skip_reasons["dropped_dataset"] += 1
            continue
        if t.get("truncated"):
            skip_reasons["truncated"] += 1
            continue
        if t.get("correct") is None:
            skip_reasons["unlabeled"] += 1
            continue
        kept.append(t)

    print(f"\nKept: {len(kept):,}  Skipped: {sum(skip_reasons.values()):,}")
    for reason, n in sorted(skip_reasons.items()):
        print(f"  {reason}: {n:,}")

    by_ds    = defaultdict(lambda: Counter())
    by_model = defaultdict(lambda: Counter())
    for t in kept:
        ds    = t["dataset"]
        model = t["model"]
        label = t["correct"]
        by_ds[ds]["total"]    += 1
        by_ds[ds]["correct"]  += int(label)
        by_model[model]["total"]   += 1
        by_model[model]["correct"] += int(label)

    print(f"\n{'dataset':<20} {'total':>7} {'correct%':>9}")
    for ds in sorted(by_ds):
        d = by_ds[ds]
        print(f"  {ds:<18} {d['total']:>7} {d['correct']/d['total']:>9.1%}")

    print(f"\n{'model':<30} {'total':>7} {'correct%':>9}")
    for m in sorted(by_model):
        d = by_model[m]
        print(f"  {m:<28} {d['total']:>7} {d['correct']/d['total']:>9.1%}")

    n_correct = sum(t["correct"] for t in kept)
    print(f"\n{'='*50}")
    print(f"FINAL DATASET")
    print(f"  Total traces   : {len(kept):,}")
    print(f"  Models         : {len(by_model)}")
    print(f"  Datasets       : {len(by_ds)}")
    print(f"  Overall correct: {n_correct/len(kept):.1%}")

    opener_out = gzip.open if args.output.endswith(".gz") else open
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with opener_out(args.output, "wt") as fo:
        for t in kept:
            fo.write(json.dumps(t) + "\n")

    summary = {
        "n_total":          len(kept),
        "n_models":         len(by_model),
        "n_datasets":       len(by_ds),
        "pct_correct":      n_correct / len(kept),
        "dropped_models":   sorted(dropped_models),
        "dropped_datasets": sorted(dropped_datasets),
        "skip_reasons":     dict(skip_reasons),
        "per_dataset":      {ds: dict(d) for ds, d in by_ds.items()},
        "per_model":        {m:  dict(d) for m,  d in by_model.items()},
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {args.output}")
    print(f"Wrote: {args.summary}")


if __name__ == "__main__":
    main()
