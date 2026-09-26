"""Export the learned pivot dictionary for inference on new traces."""

import argparse
import json
from pathlib import Path

from reasonops.paths import DATA_DIR, SPANS_FILE

PIVOT_FILE = DATA_DIR / "operators/pivot_map.json"


def build_pivots(spans):
    pivots = {}
    with Path(spans).open() as stream:
        for line in stream:
            row = json.loads(line)
            phrase, label = row.get("ngram"), row.get("cluster", -1)
            if not phrase or label is None or label < 0:
                continue
            if phrase in pivots and pivots[phrase] != label:
                raise ValueError(f"Inconsistent cluster assignment for {phrase!r}")
            pivots[phrase] = int(label)
    if not pivots:
        raise ValueError("No labeled pivots found in the span file")
    return pivots


def load_pivots(path=None):
    path = Path(path) if path else PIVOT_FILE
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, default=SPANS_FILE)
    parser.add_argument("--out", type=Path, default=PIVOT_FILE)
    args = parser.parse_args()
    pivots = build_pivots(args.spans)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pivots, indent=2) + "\n")
    print(f"Wrote {len(pivots)} pivots to {args.out}")


if __name__ == "__main__":
    main()
