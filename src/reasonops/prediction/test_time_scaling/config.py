"""Configuration for test-time selection and restart experiments."""

import json
import os

from reasonops.paths import OUTPUT_DIR

HERE = OUTPUT_DIR / "test_time_scaling"
MODELS = ["qwen3-30b-thinking", "kimi-k2.5", "r1-0528", "r1-distill-llama-70b"]
DATASETS = {"aime": 16, "livecodebench": 16}
BUDGET = int(os.environ.get("TTS_BUDGET", 160000))
ATTEMPT_CAP = 32000
MIN_START = 6000
CHECK_CHARS = 2000
TEMPERATURE = 0.7
CONCURRENCY = 64
TOKENIZERS = {
    "qwen3-30b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "kimi-k2.5": "moonshotai/Kimi-K2-Thinking",
    "r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}
PROBLEMS_JSON = HERE / "problems.json"


def problems():
    if not PROBLEMS_JSON.exists():
        from reasonops.data.benchmarks import load_dataset_problems

        out = {ds: load_dataset_problems(ds, n=n) for ds, n in DATASETS.items()}
        HERE.mkdir(parents=True, exist_ok=True)
        PROBLEMS_JSON.write_text(json.dumps(out))
    p = json.loads(PROBLEMS_JSON.read_text())
    return {ds: rows[:1] for ds, rows in p.items()} if os.environ.get("TTS_PILOT") else p


def held_out_ids():
    problems()
    all_problems = json.loads(PROBLEMS_JSON.read_text())
    return {str(p["problem_id"]) for rows in all_problems.values() for p in rows}
