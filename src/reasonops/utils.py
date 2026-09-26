"""Shared constants, display mappings, and helper functions for ReasonOps.

All pipeline modules import from here to avoid duplication.
"""
import gzip
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

# ── Corpus constants ──────────────────────────────────────────────────────────

EXCLUDE_DATASETS: frozenset = frozenset({"bbh", "humaneval", "arc_agi2"})
N_OPERATORS: int = 7
N_FOLDS: int = 5
SEED: int = 42
FRACS: tuple = (0.1, 0.25, 0.5, 0.75, 1.0)

# ── Model / dataset display mappings ─────────────────────────────────────────

MODEL_ORDER: list = [
    "qwq-32b", "qwen3-30b-thinking", "qwen3-235b-thinking",
    "r1-distill-qwen-32b", "r1-distill-llama-70b", "r1-0528",
    "claude-haiku", "claude-sonnet",
    "grok-3-mini", "nemotron-49b", "kimi-k2.5", "kimi-k2",
]

MODEL_DISPLAY: dict = {
    "qwq-32b":              "QwQ-32B",
    "qwen3-30b-thinking":   "Qwen3-30B",
    "qwen3-235b-thinking":  "Qwen3-235B",
    "r1-distill-qwen-32b":  "R1-Distill-Q32",
    "r1-distill-llama-70b": "R1-Distill-L70",
    "r1-0528":              "R1-0528",
    "claude-haiku":         "Haiku",
    "claude-sonnet":        "Sonnet",
    "grok-3-mini":          "Grok-3-Mini",
    "nemotron-49b":         "Nemotron-49B",
    "kimi-k2.5":            "Kimi-K2.5",
    "kimi-k2":              "Kimi-K2",
}

MODEL_DISPLAY_FULL: dict = {
    "qwq-32b":              "QwQ-32B",
    "qwen3-30b-thinking":   "Qwen3-30B-Thinking",
    "qwen3-235b-thinking":  "Qwen3-235B-Thinking",
    "r1-0528":              "DeepSeek R1-0528",
    "r1-distill-qwen-32b":  "R1-Distill-Qwen-32B",
    "r1-distill-llama-70b": "R1-Distill-Llama-70B",
    "claude-haiku":         "Claude Haiku 4.5",
    "claude-sonnet":        "Claude Sonnet 4.5",
    "grok-3-mini":          "Grok-3-Mini",
    "nemotron-49b":         "Nemotron-Super-49B",
    "kimi-k2":              "Kimi-K2",
    "kimi-k2.5":            "Kimi-K2.5",
}

FAMILY: dict = {
    "qwq-32b":              "Qwen / QwQ",
    "qwen3-30b-thinking":   "Qwen / QwQ",
    "qwen3-235b-thinking":  "Qwen / QwQ",
    "r1-0528":              "DeepSeek R1",
    "r1-distill-qwen-32b":  "DeepSeek R1",
    "r1-distill-llama-70b": "DeepSeek R1",
    "claude-haiku":         "Claude",
    "claude-sonnet":        "Claude",
    "grok-3-mini":          "Grok",
    "nemotron-49b":         "Nemotron",
    "kimi-k2":              "Kimi",
    "kimi-k2.5":            "Kimi",
}

DS_DISPLAY: dict = {
    "aime":          "AIME",
    "math":          "MATH",
    "gpqa":          "GPQA",
    "livecodebench": "LiveCode",
    "humaneval":     "HumanEval",
    "mmlu_pro":      "MMLU-Pro",
    "arc_challenge": "ARC-C",
    "bbh":           "BBH",
}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path) -> list:
    """Load a (optionally gzip-compressed) JSONL file into a list of dicts."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ── Cross-validation ──────────────────────────────────────────────────────────

def problem_kfold_splits(pids: np.ndarray, n_folds: int = N_FOLDS,
                         seed: int = SEED) -> list:
    """Return (train_idx, test_idx) pairs split at the problem level.

    All traces for a given problem ID are assigned to the same fold,
    preventing leakage from multi-model evaluation of the same problem.
    """
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)
    rng.shuffle(unique_pids)
    splits = []
    for fold in range(n_folds):
        te_pids = set(unique_pids[fold::n_folds])
        tr_idx = np.where(~np.isin(pids, list(te_pids)))[0]
        te_idx = np.where( np.isin(pids, list(te_pids)))[0]
        splits.append((tr_idx, te_idx))
    return splits


# ── Metrics ───────────────────────────────────────────────────────────────────

def wp_auc(y, probs, pids) -> float:
    """Within-problem AUC: average per-problem AUC over problems with both labels."""
    groups = defaultdict(list)
    for pid, yt, yp in zip(pids, y, probs):
        groups[pid].append((yt, yp))
    aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
            for v in groups.values() if len(set(p[0] for p in v)) == 2]
    return float(np.mean(aucs)) if aucs else float("nan")


# ── Classifiers ───────────────────────────────────────────────────────────────

def make_clf(seed: int = SEED):
    """Binary XGBoost classifier with paper hyperparameters."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)


def make_multiclass_clf(n_classes: int, seed: int = SEED):
    """Multi-class XGBoost classifier with paper hyperparameters."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=seed, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
