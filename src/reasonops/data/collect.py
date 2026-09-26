"""Collect per-trace JSON files into a corpus with stable trace IDs."""

import argparse
import gzip
import json
from pathlib import Path


def records(root):
    root = Path(root)
    seen = set()
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        parts = path.relative_to(root).parts
        if len(parts) == 4 and parts[0].startswith("sample_"):
            sample = int(parts[0].split("_")[1])
            model, dataset = parts[1:3]
        elif len(parts) == 3:
            sample = 1
            model, dataset = parts[:2]
        else:
            continue
        row = json.loads(path.read_text())
        problem_id = str(row.get("problem_id", path.stem))
        trace_id = f"{model}__{dataset}__{problem_id}__s{sample}"
        if trace_id in seen:
            raise ValueError(f"Duplicate trace ID: {trace_id}")
        seen.add(trace_id)
        yield {
            **row,
            "trace_id": trace_id,
            "model": model,
            "dataset": dataset,
            "problem_id": problem_id,
            "sample": sample,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    count = 0
    with opener(args.output, "wt") as f:
        for row in records(args.traces):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} traces to {args.output}")


if __name__ == "__main__":
    main()
