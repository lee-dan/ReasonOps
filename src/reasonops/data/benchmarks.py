"""
Dataset loaders for multi-model inference.

Each loader returns: list of {problem_id, problem, answer, ...}

Sampling is reproducible via seed — same seed = same 100 problems every run.
"""

import random

# Default sample sizes (None = use full dataset)
DATASET_N_DEFAULT = {
    "math":          100,
    "gpqa":          100,
    "aime":          None,   # 90 total — just do all
    "livecodebench": 100,
    "hle":           100,
    "mmlu_pro":      100,
    "arc_challenge":  100,
    "bbh":           100,
    "humaneval":     None,   # 164 total — just do all
    "arc_agi2":      100,
}


def load_dataset_problems(name: str, n: int = None, seed: int = 42) -> list[dict]:
    """
    Load benchmark problems. Returns list of dicts with at minimum:
      problem_id: str
      problem:    str  (the question text, pre-formatted for the prompt template)
      answer:     str  (ground truth for verification)
    """
    loaders = {
        "math":          _load_math,
        "gpqa":          _load_gpqa,
        "aime":          _load_aime,
        "livecodebench": _load_livecodebench,
        "hle":           _load_hle,
        "mmlu_pro":      _load_mmlu_pro,
        "arc_challenge":  _load_arc_challenge,
        "bbh":           _load_bbh,
        "humaneval":     _load_humaneval,
        "arc_agi2":      _load_arc_agi2,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(loaders)}")

    problems = loaders[name]()

    default_n = DATASET_N_DEFAULT.get(name)
    target_n = n if n is not None else default_n
    if target_n is not None and len(problems) > target_n:
        rng = random.Random(seed)
        problems = rng.sample(problems, target_n)

    return sorted(problems, key=lambda p: p["problem_id"])


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_math() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "problem_id": str(row["unique_id"]),
            "problem":    row["problem"],
            "answer":     str(row["answer"]),
            "subject":    row.get("subject", ""),
            "level":      row.get("level", ""),
        }
        for row in ds
    ]


def _load_gpqa() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond",
                      split="train", trust_remote_code=True)
    problems = []
    for i, row in enumerate(ds):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Shuffle options deterministically per problem
        rng = random.Random(i)
        rng.shuffle(options)
        correct_idx = options.index(row["Correct Answer"])
        label = chr(ord("A") + correct_idx)

        choices_str = "\n".join(
            f"({chr(ord('A') + j)}) {opt}" for j, opt in enumerate(options)
        )
        problems.append({
            "problem_id": f"gpqa_{i:04d}",
            "problem":    f"{row['Question']}\n\n{choices_str}",
            "answer":     label,
        })
    return problems


def _load_aime() -> list[dict]:
    from datasets import load_dataset
    # AIME 2024 — 90 problems total
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    return [
        {
            "problem_id": f"aime_{i:03d}",
            "problem":    row["problem"],
            "answer":     str(row["answer"]),
            "url":        row.get("url", ""),
        }
        for i, row in enumerate(ds)
    ]


def _load_livecodebench() -> list[dict]:
    import json as _json
    from pathlib import Path as _Path

    # datasets>=4.0 dropped custom script support; fall back to pre-exported JSONL.
    # Path can be overridden with $REASONOPS_LIVECODEBENCH_JSONL.
    import os as _os
    _JSONL = _Path(_os.environ.get(
        "REASONOPS_LIVECODEBENCH_JSONL",
        "data/livecodebench_test.jsonl",
    ))

    def _parse(rows):
        return [
            {
                "problem_id":   str(row.get("question_id", f"lcb_{i:04d}")),
                "problem":      row["question_content"],
                "answer":       "",
                "test_cases":   row.get("public_test_cases", []),
                "starter_code": row.get("starter_code", ""),
            }
            for i, row in enumerate(rows)
        ]

    if _JSONL.exists():
        rows = [_json.loads(l) for l in _JSONL.read_text().splitlines() if l.strip()]
        return _parse(rows)

    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite",
                      split="test", trust_remote_code=True)
    return _parse(list(ds))


# ── NEW DATASETS ─────────────────────────────────────────────────────────────

def _load_hle() -> list[dict]:
    """Humanity's Last Exam — 2500 expert-level questions, mixed format."""
    from datasets import load_dataset
    ds = load_dataset("cais/hle", split="test")
    problems = []
    for i, row in enumerate(ds):
        answer_type = "multiple_choice" if row.get("answer_type") == "multiple_choice" else "exact_match"
        problems.append({
            "problem_id":  f"hle_{i:04d}",
            "problem":     row["question"],
            "answer":      str(row["answer"]),
            "answer_type": answer_type,
            "subject":     row.get("subject", ""),
        })
    return problems


def _load_mmlu_pro() -> list[dict]:
    """MMLU-Pro humanities subset — 10-choice MCQ."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    # Filter to humanities/social science categories
    humanities = {"philosophy", "history", "law", "psychology", "economics",
                  "business", "other", "political_science"}
    problems = []
    for i, row in enumerate(ds):
        cat = row.get("category", "").lower().replace(" ", "_")
        if cat not in humanities:
            continue
        options = row.get("options", [])
        if not options:
            continue
        choices_str = "\n".join(
            f"({chr(65 + j)}) {opt}" for j, opt in enumerate(options)
        )
        problems.append({
            "problem_id": f"mmlu_pro_{i:05d}",
            "problem":    f"{row['question']}\n\n{choices_str}",
            "answer":     row["answer"],
            "category":   row.get("category", ""),
        })
    return problems


def _load_arc_challenge() -> list[dict]:
    """ARC-Challenge — science reasoning MCQ."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    problems = []
    for i, row in enumerate(ds):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choices_str = "\n".join(
            f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)
        )
        problems.append({
            "problem_id": f"arc_{i:04d}",
            "problem":    f"{row['question']}\n\n{choices_str}",
            "answer":     row["answerKey"],
        })
    return problems


def _load_bbh() -> list[dict]:
    """Big Bench Hard — diverse reasoning tasks."""
    from datasets import load_dataset
    # Load a subset of BBH tasks
    tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "formal_fallacies", "geometric_shapes",
        "hyperbaton", "logical_deduction_five_objects", "movie_recommendation",
        "navigate", "penguins_in_a_table", "reasoning_about_colored_objects",
        "snarks", "sports_understanding", "temporal_sequences",
        "tracking_shuffled_objects_three_objects", "web_of_lies",
    ]
    problems = []
    for task_name in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task_name, split="test",
                              trust_remote_code=True)
            for i, row in enumerate(ds):
                problems.append({
                    "problem_id": f"bbh_{task_name}_{i:03d}",
                    "problem":    row["input"],
                    "answer":     row["target"],
                    "task":       task_name,
                })
        except Exception:
            continue
    return problems


def _load_humaneval() -> list[dict]:
    """HumanEval — 164 Python code completion problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "problem_id":   row["task_id"],
            "problem":      row["prompt"],
            "answer":       "",
            "test":         row["test"],
            "entry_point":  row["entry_point"],
            "canonical":    row.get("canonical_solution", ""),
        }
        for row in ds
    ]


def _load_arc_agi2() -> list[dict]:
    """ARC-AGI-2 — abstract visual reasoning (grid patterns)."""
    import json as _json
    from pathlib import Path as _Path

    # Try a local clone first (path overridable via $REASONOPS_ARC_AGI2_DIR).
    import os as _os
    arc_dir = _Path(_os.environ.get(
        "REASONOPS_ARC_AGI2_DIR",
        "data/arc-agi-2",
    ))

    if not arc_dir.exists():
        # Fall back to HuggingFace download.
        from datasets import load_dataset
        ds = load_dataset("arcprize/arc_agi_2_human_testing", split="test")
        problems = []
        for i, row in enumerate(ds):
            train_examples = ""
            for j, (inp, out) in enumerate(zip(row.get("train_input", []),
                                                row.get("train_output", []))):
                train_examples += f"Example {j+1}:\nInput: {_json.dumps(inp)}\nOutput: {_json.dumps(out)}\n\n"
            test_input = row.get("test_input", [[]])
            problems.append({
                "problem_id": f"arc_agi2_{i:04d}",
                "problem":    f"{train_examples}Test input: {_json.dumps(test_input)}",
                "answer":     row.get("test_output", []),
            })
        return problems

    # Load from local JSON files
    eval_dir = arc_dir / "data" / "evaluation"
    problems = []
    if eval_dir.exists():
        for fp in sorted(eval_dir.glob("*.json")):
            task = _json.loads(fp.read_text())
            train_examples = ""
            for j, ex in enumerate(task.get("train", [])):
                train_examples += (f"Example {j+1}:\n"
                                   f"Input: {_json.dumps(ex['input'])}\n"
                                   f"Output: {_json.dumps(ex['output'])}\n\n")
            for k, test in enumerate(task.get("test", [])):
                problems.append({
                    "problem_id": f"arc_agi2_{fp.stem}_{k}",
                    "problem":    f"{train_examples}Test input: {_json.dumps(test['input'])}",
                    "answer":     test.get("output", []),
                })
    return problems


