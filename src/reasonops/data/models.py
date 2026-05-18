"""
Model registry for multi-model inference.

12 thinking LLMs across 5 families. All expose raw chain-of-thought tokens.

reasoning_format:
  "effort"    → {"reasoning": {"effort": "high"}}   — xAI
  "max_tokens"→ {"reasoning": {"max_tokens": N}}     — Qwen thinking
  "default"   → {"reasoning": {}}                   — DeepSeek, NVIDIA, Moonshot, QwQ
  "anthropic" → {"thinking": {"type":"enabled",...}} — Claude direct API (temp must = 1)
"""

MODEL_REGISTRY = {
    # ═══ Qwen (3) ═══════════════════════════════════════════════════
    "qwq-32b": {
        "display_name": "QwQ-32B",
        "or_model_id":  "qwen/qwq-32b",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "default",
        "think_prefix": False,
    },
    "qwen3-235b-thinking": {
        "display_name": "Qwen3-235B-Thinking",
        "or_model_id":  "qwen/qwen3-235b-a22b-thinking-2507",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "max_tokens",
        "think_prefix": False,
    },
    "qwen3-30b-thinking": {
        "display_name": "Qwen3-30B-Thinking",
        "or_model_id":  "qwen/qwen3-30b-a3b-thinking-2507",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "max_tokens",
        "think_prefix": False,
    },

    # ═══ DeepSeek (3) ════════════════════════════════════════════════
    "r1-0528": {
        "display_name": "DeepSeek-R1-0528",
        "or_model_id":  "deepseek/deepseek-r1-0528",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "default",
        "think_prefix": False,
    },
    "r1-distill-llama-70b": {
        "display_name": "R1-Distill-Llama-70B",
        "or_model_id":  "deepseek/deepseek-r1-distill-llama-70b",
        "backend":      "openrouter",
        "max_new_tokens": 16384,
        "reasoning_format": "default",
        "think_prefix": False,
    },
    "r1-distill-qwen-32b": {
        "display_name": "R1-Distill-Qwen-32B",
        "or_model_id":  "deepseek/deepseek-r1-distill-qwen-32b",
        "backend":      "openrouter",
        "max_new_tokens": 27000,
        "reasoning_format": "default",
        "think_prefix": False,
    },

    # ═══ Anthropic (2) — direct API ════════════════════════════════
    "claude-sonnet": {
        "display_name": "Claude-Sonnet-4.5",
        "or_model_id":  "claude-sonnet-4-20250514",
        "backend":      "anthropic",
        "max_new_tokens": 64000,
        "reasoning_format": "anthropic",
        "think_prefix": False,
    },
    "claude-haiku": {
        "display_name": "Claude-Haiku-4.5",
        "or_model_id":  "claude-haiku-4-5-20251001",
        "backend":      "anthropic",
        "max_new_tokens": 64000,
        "reasoning_format": "anthropic",
        "think_prefix": False,
    },

    # ═══ xAI (1) ═════════════════════════════════════════════════════
    "grok-3-mini": {
        "display_name": "Grok-3-Mini",
        "or_model_id":  "x-ai/grok-3-mini",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "effort",
        "think_prefix": False,
    },

    # ═══ NVIDIA (1) ══════════════════════════════════════════════════
    "nemotron-49b": {
        "display_name": "Nemotron-Super-49B",
        "or_model_id":  "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "default",
        "think_prefix": False,
    },

    # ═══ Moonshot (2) ════════════════════════════════════════════════
    "kimi-k2": {
        "display_name": "Kimi-K2-Thinking",
        "or_model_id":  "moonshotai/kimi-k2-thinking",
        "backend":      "openrouter",
        "max_new_tokens": 65000,
        "reasoning_format": "default",
        "think_prefix": False,
    },
    "kimi-k2.5": {
        "display_name": "Kimi-K2.5",
        "or_model_id":  "moonshotai/kimi-k2.5",
        "backend":      "openrouter",
        "max_new_tokens": 65535,
        "reasoning_format": "default",
        "think_prefix": False,
    },
}

ALL_MODELS = list(MODEL_REGISTRY.keys())

ALL_DATASETS = [
    "math", "gpqa", "aime", "livecodebench",
    "mmlu_pro", "arc_challenge", "bbh", "humaneval",
]

PROMPT_TEMPLATES = {
    "math": (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\boxed{{}}.\n\n{problem}"
    ),
    "gpqa": (
        "Answer the following multiple choice question. Think carefully, "
        "then state your answer as 'The answer is (X)' where X is the letter.\n\n{problem}"
    ),
    "aime": (
        "Solve the following AIME competition problem. "
        "Your final answer must be an integer from 0 to 999. "
        "Put your answer inside \\boxed{{}}.\n\n{problem}"
    ),
    "livecodebench": (
        "Solve the following programming problem. "
        "Provide a complete, runnable Python solution in a ```python code block.\n\n{problem}"
    ),
    "mmlu_pro": (
        "Answer the following multiple choice question. Think step by step, "
        "then state your answer as 'The answer is (X)' where X is the letter.\n\n{problem}"
    ),
    "arc_challenge": (
        "Answer the following science question. "
        "State your answer as 'The answer is (X)' where X is the letter.\n\n{problem}"
    ),
    "bbh": (
        "Answer the following question. Think step by step. "
        "End with 'The answer is [your answer]'.\n\n{problem}"
    ),
    "humaneval": (
        "Complete the following Python function. "
        "Provide only the function body in a ```python code block.\n\n{problem}"
    ),
}
