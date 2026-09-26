"""Data locations shared by analysis commands."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("REASONOPS_DATA", "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("REASONOPS_OUTPUT", str(DATA_DIR))).expanduser().resolve()
SPANS_FILE = (
    Path(
        os.environ.get(
            "REASONOPS_SPANS", str(DATA_DIR / "operators/discovered_k7/spans_clustered.jsonl")
        )
    )
    .expanduser()
    .resolve()
)
