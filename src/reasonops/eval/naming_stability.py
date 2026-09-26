#!/usr/bin/env python3
"""
Operator validation study — three analyses in one script.

1. CENTROID SAMPLING: For each of the k operator clusters, find the 30 spans
   whose anchor first-word lies closest to the cluster centroid in UMAP-cluster
   space. Save as a CSV.

2. LLM KAPPA STUDY: Sample 20 spans per cluster (200 total), ask
   claude-haiku-4-5-20251001 to independently assign each to one of our operator
   categories, then compute Cohen's kappa and a confusion matrix PNG.

3. RST ALIGNMENT: For the same 200 spans, ask the LLM to also assign an RST
   relation type. Build a 10×16 operator-RST alignment matrix (row-normalized)
   and save as a PNG.

Inputs (defaults assume the discovery output layout under data/operators/):
  --input      data/operators/discovered/spans_clustered.jsonl
  --centroids  data/operators/discovered/umap_cluster.npy
               (vocab order:    data/operators/discovered/operator_vocab.txt)
               (phrase→cluster: data/operators/discovered/umap_viz.jsonl)
  --names      data/operators/discovered/cluster_names.json
  --output_dir data/operators/validation/

Usage:
    python -m reasonops.eval.naming_stability \\
        --input      .../spans_clustered.jsonl \\
        --viz        .../umap_viz.jsonl \\
        --centroids  .../umap_cluster.npy \\
        --vocab      .../operator_vocab.txt \\
        --names      .../cluster_names.json \\
        --output_dir .../validation/
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── RST relation taxonomy ─────────────────────────────────────────────────────
# RST-DT 16 coarse-grained semantic classes (Carlson & Marcu 2001, derived from
# Mann & Thompson 1988).  We drop the two structural pseudo-relations
# (Textual-Organization, Same-Unit) as they do not apply to span-level reasoning.
# Reference: Carlson, Marcu & Okurowski (2001), "Building a Discourse-Tagged
# Corpus in the Framework of Rhetorical Structure Theory."

RST_RELATIONS = [
    "Attribution",    # satellite attributes source/belief to nucleus agent
    "Background",     # satellite provides context that increases comprehensibility
    "Cause",          # volitional or non-volitional cause of nucleus situation
    "Comparison",     # two or more comparable nuclei, differences highlighted
    "Condition",      # realization of nucleus depends on satellite condition
    "Contrast",       # two or more nuclei are contrasted; no clear comparison
    "Elaboration",    # satellite provides additional detail about nucleus
    "Enablement",     # satellite increases reader's ability to perform nucleus action
    "Evaluation",     # satellite assesses the quality/value of the nucleus
    "Explanation",    # satellite provides reason or explanation for nucleus
    "Joint",          # nuclei are related only by shared context (list/conjunction)
    "Manner-Means",   # satellite describes how or by what means nucleus was achieved
    "Summary",        # satellite restates nucleus more concisely
    "Temporal",       # nucleus and satellite are related by time order
    "Topic-Change",   # topic shifts between nucleus and satellite
    "Topic-Comment",  # satellite comments on or reacts to nucleus topic
]


# ── Operator descriptions used in the LLM prompt ─────────────────────────────
# These are embedded into the prompt and come from the cluster_names.json
# descriptions. We also carry a short fallback description per canonical name
# for robustness when a run uses different cluster orderings.

_CANONICAL_DESCRIPTIONS = {
    "CONFIRMATION":  "Affirming or restating that something is correct or established.",
    "ELABORATION":   "Expanding detail on the current point with specifics or sub-steps.",
    "SPECULATION":   "Entertaining a possible interpretation or alternative before deciding.",
    "PROGRESSION":   "Moving to the next concrete computational or logical step.",
    "INITIATION":    "Setting up the problem — identifying unknowns, givens, and goals.",
    "ACCEPTANCE":    "Acknowledging a result or intermediate finding and continuing.",
    "DETERMINATION": "Pinning down exactly what is now established — committing explicitly.",
    "DEDUCTION":     "Stating a logical conclusion that follows from prior steps.",
    "INTERRUPTION":  "Pausing to check for errors, edge cases, or incorrect assumptions.",
    "REFRAMING":     "Abandoning the current approach to try a different angle.",
}


# ── Span text helper ──────────────────────────────────────────────────────────

def span_text(span: dict, max_len: int = 300) -> str:
    """Return anchor + first body sentence, capped at max_len chars."""
    text = span.get("anchor", "").strip()
    body = span.get("body", [])
    if body:
        text += " " + body[0].strip()
    # Truncate at first sentence boundary if available
    m = re.search(r"[.!?]\s", text)
    if m and m.start() > 30:
        text = text[: m.start() + 1]
    return text[:max_len].strip()


# ── Load artifacts ────────────────────────────────────────────────────────────

def load_names(path: str) -> dict[int, dict]:
    """Load cluster_names.json → {cluster_id: {name, description, exemplars}}."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_viz(path: str) -> tuple[list[str], list[int]]:
    """Load umap_viz.jsonl → (phrases, cluster_ids)."""
    phrases, clusters = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            phrases.append(r["phrase"])
            clusters.append(r["cluster"])
    return phrases, clusters


def load_vocab_order(path: str) -> list[str]:
    """Load operator_vocab.txt (one word per line, alphabetically sorted)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_spans_by_cluster(spans_path: str, valid_cluster_ids: set) -> dict[int, list[dict]]:
    """Stream spans_clustered.jsonl and bucket by cluster id."""
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    with open(spans_path) as f:
        for line in f:
            r = json.loads(line)
            c = r.get("cluster", -2)
            if c in valid_cluster_ids and r.get("anchor", "").strip():
                by_cluster[c].append(r)
    return dict(by_cluster)


# ── Part 1: centroid-nearest spans ───────────────────────────────────────────

def sample_centroid_spans(
    centroids_path: str,
    vocab_path: str,
    viz_path: str,
    spans_path: str,
    names: dict[int, dict],
    n_per_cluster: int = 30,
) -> list[dict]:
    """
    For each cluster, find the n_per_cluster first-words closest to the cluster
    centroid in UMAP-cluster space, then collect one span per chosen first-word.

    Returns a list of dicts: {cluster_id, operator_name, first_word, anchor,
                               body_preview, distance_to_centroid}.
    """
    print("\n=== Part 1: Centroid-nearest span sampling ===")

    # Load the UMAP cluster embedding (one row per vocab word, in vocab order)
    emb_all = np.load(centroids_path)          # shape: (V, D)
    vocab = load_vocab_order(vocab_path)        # length V

    # Load viz to get phrase→cluster mapping
    phrases_viz, clusters_viz = load_viz(viz_path)
    phrase_to_cluster = dict(zip(phrases_viz, clusters_viz))

    # Build {phrase: embedding_row} — vocab order matches emb_all row order
    if len(vocab) != len(emb_all):
        raise ValueError(
            f"vocab length ({len(vocab)}) != embedding rows ({len(emb_all)}). "
            "Make sure --vocab and --centroids come from the same operator discovery run."
        )
    vocab_to_emb = {w: emb_all[i] for i, w in enumerate(vocab)}

    cluster_ids = sorted(names.keys())

    # Compute per-cluster centroids from phrase embeddings
    cluster_centroids: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        phrase_idxs = [i for i, cl in enumerate(clusters_viz) if cl == cid
                       and phrases_viz[i] in vocab_to_emb]
        if not phrase_idxs:
            print(f"  [warn] cluster {cid} has no vocab phrases — skipping centroid")
            continue
        vecs = np.stack([vocab_to_emb[phrases_viz[i]] for i in phrase_idxs])
        cluster_centroids[cid] = vecs.mean(axis=0)

    # For each cluster, rank phrases by distance to centroid
    top_phrases: dict[int, list[str]] = {}
    for cid, centroid in cluster_centroids.items():
        phrase_idxs = [i for i, cl in enumerate(clusters_viz) if cl == cid
                       and phrases_viz[i] in vocab_to_emb]
        vecs = np.stack([vocab_to_emb[phrases_viz[i]] for i in phrase_idxs])
        dists = np.linalg.norm(vecs - centroid, axis=1)
        sorted_idx = np.argsort(dists)[:n_per_cluster]
        top_phrases[cid] = [phrases_viz[phrase_idxs[j]] for j in sorted_idx]

    # Collect one span per target first-word (first match wins)
    needed: dict[int, set] = {cid: set(phrases) for cid, phrases in top_phrases.items()}
    collected: dict[int, dict[str, dict]] = {cid: {} for cid in cluster_ids}

    print("  Streaming spans to collect centroid-nearest examples...")
    with open(spans_path) as f:
        for line in f:
            if all(len(v) >= len(top_phrases.get(cid, [])) for cid, v in collected.items()):
                break
            r = json.loads(line)
            c = r.get("cluster", -2)
            if c not in needed:
                continue
            anchor = r.get("anchor", "").strip()
            first = anchor.split()[0].lower().rstrip(".,!?;:'\"") if anchor.split() else ""
            if first in needed[c] and first not in collected[c] and anchor:
                collected[c][first] = r

    # Build output records (in centroid-distance order)
    rows = []
    for cid in cluster_ids:
        if cid not in cluster_centroids:
            continue
        centroid = cluster_centroids[cid]
        op_name = names[cid]["name"] if isinstance(names[cid], dict) else str(names[cid])
        for phrase in top_phrases[cid]:
            if phrase not in collected[cid]:
                continue
            r = collected[cid][phrase]
            emb = vocab_to_emb[phrase]
            dist = float(np.linalg.norm(emb - centroid))
            body = r.get("body", [])
            rows.append({
                "cluster_id":           cid,
                "operator_name":        op_name,
                "first_word":           phrase,
                "anchor":               r.get("anchor", ""),
                "body_preview":         " ".join(body[:2])[:120],
                "distance_to_centroid": round(dist, 5),
                "trace_id":             r.get("trace_id", ""),
                "span_idx":             r.get("span_idx", 0),
            })

    print(f"  Collected {len(rows)} centroid-nearest spans across "
          f"{len([c for c in collected if collected[c]])} clusters")
    return rows


# ── Rate-limited Anthropic client ─────────────────────────────────────────────

_last_call_time: float = 0.0

def _sleep_for_rate_limit(min_interval: float = 0.1):
    global _last_call_time
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.monotonic()


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
)
def _call_api(client: anthropic.Anthropic, model: str, prompt: str,
              rate_limit_interval: float = 0.1) -> str:
    _sleep_for_rate_limit(min_interval=rate_limit_interval)
    msg = client.messages.create(
        model=model,
        max_tokens=32,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── Build LLM prompts ─────────────────────────────────────────────────────────

def _operator_list_block(names: dict[int, dict]) -> str:
    """Format the 10 operators with one-line descriptions for the LLM prompt."""
    lines = []
    for cid in sorted(names.keys()):
        entry = names[cid]
        name = entry["name"] if isinstance(entry, dict) else str(entry)
        desc = (entry.get("description", "") if isinstance(entry, dict) else "")
        if not desc:
            desc = _CANONICAL_DESCRIPTIONS.get(name, "")
        lines.append(f"  {name}: {desc}")
    return "\n".join(lines)


def build_operator_prompt(span: dict, names: dict[int, dict]) -> str:
    op_block = _operator_list_block(names)
    text = span_text(span, max_len=400)
    return (
        "You are a discourse analyst. A reasoning span is one logical step in a "
        "chain-of-thought.\n"
        "Assign this span to exactly one of these 10 operator categories:\n"
        f"{op_block}\n\n"
        f'Span: "{text}"\n\n'
        "Reply with just the operator name, nothing else."
    )


def build_rst_prompt(span: dict) -> str:
    # Include one-line definitions so the LLM applies the Carlson & Marcu (2001)
    # meanings rather than everyday interpretations of the names.
    rst_defs = "\n".join(f"  {r}" for r in RST_RELATIONS)
    text = span_text(span, max_len=400)
    return (
        "You are a discourse analyst applying Rhetorical Structure Theory (RST) "
        "as defined by Mann & Thompson (1988) and the RST Discourse Treebank "
        "(Carlson & Marcu 2001).\n"
        "Assign this reasoning span to exactly one of the 16 coarse-grained RST "
        "relation classes listed below. The relation should describe the primary "
        "rhetorical function this span serves relative to the surrounding reasoning.\n\n"
        "RST relation classes:\n"
        f"{rst_defs}\n\n"
        f'Span: "{text}"\n\n'
        "Reply with just the RST relation name (exact spelling from the list), nothing else."
    )


# ── Parse LLM response ────────────────────────────────────────────────────────

def _parse_operator(response: str, valid_names: set[str]) -> str | None:
    """Extract a valid operator name from an LLM response (case-insensitive)."""
    resp_clean = response.strip().upper()
    for name in valid_names:
        if name.upper() in resp_clean:
            return name
    return None


def _parse_rst(response: str) -> str | None:
    """Extract a valid RST relation name from an LLM response."""
    resp_clean = response.strip()
    rst_upper = {r.upper(): r for r in RST_RELATIONS}
    for token in re.split(r"[\s,;]+", resp_clean):
        token_up = token.upper().rstrip(".,!?;:'\"")
        if token_up in rst_upper:
            return rst_upper[token_up]
    return None


# ── Part 2: LLM kappa study ───────────────────────────────────────────────────

def run_kappa_study(
    test_spans: list[tuple[dict, int]],
    names: dict[int, dict],
    client: anthropic.Anthropic,
    model: str,
    rate_limit_interval: float = 0.1,
) -> tuple[list[int], list[int | None]]:
    """
    Call LLM to classify each span; return (y_true, y_pred_as_cluster_ids).
    """
    print(f"\n=== Part 2: LLM kappa study ({len(test_spans)} spans) ===")
    cluster_ids = sorted(names.keys())
    name_to_id = {
        (entry["name"] if isinstance(entry, dict) else str(entry)): cid
        for cid, entry in names.items()
    }
    valid_names = set(name_to_id.keys())

    y_true: list[int] = []
    y_pred: list[int | None] = []

    for i, (span, true_c) in enumerate(test_spans):
        prompt = build_operator_prompt(span, names)
        try:
            response = _call_api(client, model, prompt,
                                 rate_limit_interval=rate_limit_interval)
            parsed_name = _parse_operator(response, valid_names)
            pred_c = name_to_id.get(parsed_name) if parsed_name else None
        except Exception as e:
            print(f"  [warn] span {i}: API error — {e}")
            pred_c = None

        y_true.append(true_c)
        y_pred.append(pred_c)

        if (i + 1) % 20 == 0:
            n_ok = sum(1 for p in y_pred if p is not None)
            print(f"  {i+1}/{len(test_spans)} classified  ({n_ok} valid responses)")

    return y_true, y_pred


# ── Part 3: RST alignment ─────────────────────────────────────────────────────

def run_rst_study(
    test_spans: list[tuple[dict, int]],
    client: anthropic.Anthropic,
    model: str,
    rate_limit_interval: float = 0.1,
) -> list[str | None]:
    """Ask LLM to assign an RST relation to each span. Returns list of RST labels."""
    print(f"\n=== Part 3: RST alignment study ({len(test_spans)} spans) ===")

    rst_labels: list[str | None] = []
    for i, (span, _) in enumerate(test_spans):
        prompt = build_rst_prompt(span)
        try:
            response = _call_api(client, model, prompt,
                                 rate_limit_interval=rate_limit_interval)
            parsed = _parse_rst(response)
        except Exception as e:
            print(f"  [warn] span {i}: API error — {e}")
            parsed = None

        rst_labels.append(parsed)

        if (i + 1) % 20 == 0:
            n_ok = sum(1 for r in rst_labels if r is not None)
            print(f"  {i+1}/{len(test_spans)} RST-labeled  ({n_ok} valid)")

    return rst_labels


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    op_names: list[str],
    out_path: Path,
    kappa: float,
) -> None:
    """Save confusion matrix as a PNG heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(op_names)))
    ax.set_yticks(range(len(op_names)))
    ax.set_xticklabels(op_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(op_names, fontsize=8)
    ax.set_xlabel("LLM assignment", fontsize=10)
    ax.set_ylabel("K-means assignment", fontsize=10)
    ax.set_title(f"Operator classification confusion matrix\nCohen's κ = {kappa:.3f}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(op_names)):
        for j in range(len(op_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix → {out_path}")


def plot_rst_alignment(
    alignment: np.ndarray,
    op_names: list[str],
    out_path: Path,
) -> None:
    """Save RST alignment matrix as a PNG heatmap (operators × RST relations)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(alignment, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(RST_RELATIONS)))
    ax.set_yticks(range(len(op_names)))
    ax.set_xticklabels(RST_RELATIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(op_names, fontsize=8)
    ax.set_xlabel("RST relation", fontsize=10)
    ax.set_ylabel("Operator (K-means)", fontsize=10)
    ax.set_title("RST alignment matrix\n(row-normalized: P(RST | operator))", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells with values ≥ 0.05
    for i in range(len(op_names)):
        for j in range(len(RST_RELATIONS)):
            v = alignment[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > 0.6 else "black",
                        fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved RST alignment matrix → {out_path}")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Operator validation: centroid sampling, "
                                              "LLM kappa, and RST alignment.")
    ap.add_argument("--input",       required=True,
                    help="spans_clustered.jsonl from embed_operators.py output")
    ap.add_argument("--viz",         required=True,
                    help="umap_viz.jsonl — phrase/cluster/x/y per first-word")
    ap.add_argument("--centroids",   required=True,
                    help="umap_cluster.npy — UMAP cluster-space embeddings "
                         "(one row per vocab word, in operator_vocab.txt order)")
    ap.add_argument("--vocab",       required=True,
                    help="operator_vocab.txt — one operator word per line")
    ap.add_argument("--names",       required=True,
                    help="cluster_names.json — {cluster_id: {name, description, exemplars}}")
    ap.add_argument("--output_dir",          required=True)
    ap.add_argument("--naming_model",        default="claude-haiku-4-5-20251001",
                    help="Anthropic model for LLM labeling (default: claude-haiku-4-5-20251001)")
    ap.add_argument("--n_centroid",          type=int, default=30,
                    help="Spans per cluster for centroid sampling (default 30)")
    ap.add_argument("--n_kappa",             type=int, default=20,
                    help="Spans per cluster for LLM kappa + RST study (default 20)")
    ap.add_argument("--rate_limit_interval", type=float, default=0.1,
                    help="Minimum seconds between API calls (default 0.1 = 10 req/sec)")
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--skip_llm",            action="store_true",
                    help="Skip LLM calls (useful for testing loading logic)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.skip_llm:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    # ── Load cluster names ────────────────────────────────────────────────────
    print(f"Loading cluster names from {args.names}")
    names = load_names(args.names)
    cluster_ids = sorted(names.keys())
    op_names_ordered = [
        names[cid]["name"] if isinstance(names[cid], dict) else str(names[cid])
        for cid in cluster_ids
    ]
    n_clusters = len(cluster_ids)
    print(f"  {n_clusters} operators: {', '.join(op_names_ordered)}")

    # ── Part 1: centroid-nearest spans ───────────────────────────────────────
    centroid_rows = sample_centroid_spans(
        centroids_path=args.centroids,
        vocab_path=args.vocab,
        viz_path=args.viz,
        spans_path=args.input,
        names=names,
        n_per_cluster=args.n_centroid,
    )
    write_csv(centroid_rows, out / "centroid_spans.csv")
    print(f"  Part 1 complete: {len(centroid_rows)} centroid-nearest spans saved.")

    if args.skip_llm:
        print("\n--skip_llm set. Exiting after Part 1.")
        return

    # ── Sample 200 spans for Parts 2 & 3 ─────────────────────────────────────
    print(f"\nSampling {args.n_kappa} spans/cluster for LLM studies...")
    rng = random.Random(args.seed)
    by_cluster = load_spans_by_cluster(args.input, set(cluster_ids))

    test_spans: list[tuple[dict, int]] = []
    for cid in cluster_ids:
        pool = by_cluster.get(cid, [])
        k = min(args.n_kappa, len(pool))
        if k == 0:
            print(f"  [warn] cluster {cid} ({op_names_ordered[cluster_ids.index(cid)]}) "
                  "has no spans — skipped")
            continue
        sampled = rng.sample(pool, k)
        test_spans.extend([(s, cid) for s in sampled])

    print(f"  {len(test_spans)} spans selected for LLM annotation")

    # ── Initialize Anthropic client ───────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)

    # ── Part 2: LLM kappa study ───────────────────────────────────────────────
    y_true, y_pred_raw = run_kappa_study(test_spans, names, client, args.naming_model,
                                          rate_limit_interval=args.rate_limit_interval)

    # Filter out failed predictions for kappa computation
    valid_mask = [p is not None for p in y_pred_raw]
    y_true_valid = [y for y, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [p for p, m in zip(y_pred_raw, valid_mask) if m]
    n_failed = sum(1 for m in valid_mask if not m)

    kappa = cohen_kappa_score(y_true_valid, y_pred_valid) if len(y_true_valid) > 1 else float("nan")
    acc = sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t == p) / max(len(y_true_valid), 1)

    print(f"\n  N={len(y_true_valid)} valid / {n_failed} failed")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Cohen's κ: {kappa:.4f}")

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=cluster_ids)
    plot_confusion_matrix(cm, op_names_ordered, out / "confusion_matrix.png", kappa)

    # ── Part 3: RST alignment ─────────────────────────────────────────────────
    rst_labels = run_rst_study(test_spans, client, args.naming_model,
                               rate_limit_interval=args.rate_limit_interval)

    # Build 10×16 alignment matrix (counts, then row-normalize)
    rst_index = {r: j for j, r in enumerate(RST_RELATIONS)}
    alignment_counts = np.zeros((n_clusters, len(RST_RELATIONS)), dtype=float)

    for (span, true_c), rst in zip(test_spans, rst_labels):
        if rst is None:
            continue
        row = cluster_ids.index(true_c)
        col = rst_index[rst]
        alignment_counts[row, col] += 1

    # Row-normalize
    row_sums = alignment_counts.sum(axis=1, keepdims=True)
    alignment_norm = alignment_counts / np.maximum(row_sums, 1.0)

    plot_rst_alignment(alignment_norm, op_names_ordered, out / "rst_alignment.png")

    # ── Build annotated CSV ───────────────────────────────────────────────────
    print("\nBuilding annotated CSV...")
    csv_rows = []
    for i, ((span, true_c), pred_c, rst) in enumerate(
        zip(test_spans, y_pred_raw, rst_labels)
    ):
        true_name = op_names_ordered[cluster_ids.index(true_c)]
        pred_name = None
        if pred_c is not None and pred_c in cluster_ids:
            pred_name = op_names_ordered[cluster_ids.index(pred_c)]
        csv_rows.append({
            "span_idx":          i,
            "trace_id":          span.get("trace_id", ""),
            "span_position":     span.get("span_idx", 0),
            "true_cluster_id":   true_c,
            "true_operator":     true_name,
            "llm_operator":      pred_name if pred_name else "",
            "llm_cluster_id":    pred_c if pred_c is not None else "",
            "operator_match":    int(true_c == pred_c) if pred_c is not None else "",
            "rst_relation":      rst if rst else "",
            "anchor":            span.get("anchor", "")[:200],
            "body_preview":      " ".join(span.get("body", [])[:2])[:150],
        })
    write_csv(csv_rows, out / "annotated_spans.csv")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "model": args.naming_model,
        "n_clusters": n_clusters,
        "operator_names": op_names_ordered,
        "n_spans_total": len(test_spans),
        "n_spans_valid_operator": len(y_true_valid),
        "n_spans_failed": n_failed,
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "cluster_ids": cluster_ids,
        "rst_relations": RST_RELATIONS,
        "rst_alignment_matrix": alignment_norm.tolist(),
        "rst_alignment_counts": alignment_counts.tolist(),
    }
    summary_path = out / "operator_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved summary → {summary_path}")

    # ── Print RST alignment table ─────────────────────────────────────────────
    print("\n=== RST alignment (top-2 RST per operator) ===")
    for i, op_name in enumerate(op_names_ordered):
        row = alignment_norm[i]
        top2 = np.argsort(row)[::-1][:2]
        pairs = [f"{RST_RELATIONS[j]}={row[j]:.2f}" for j in top2 if row[j] > 0]
        print(f"  {op_name:<16}: {', '.join(pairs)}")

    print(f"\nAll outputs written to {out}/")
    print(f"  centroid_spans.csv         — {len(centroid_rows)} centroid-nearest spans")
    print(f"  annotated_spans.csv        — {len(csv_rows)} spans with both LLM labels")
    print(f"  confusion_matrix.png       — kappa={kappa:.3f}")
    print(f"  rst_alignment.png          — 10×16 operator-RST matrix")
    print(f"  operator_validation_summary.json")


if __name__ == "__main__":
    main()
