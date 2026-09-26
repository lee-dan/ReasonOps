"""Generate baseline and OST-guided attempts under a completion-token budget."""

import asyncio
import json
import os
import time

import httpx

from reasonops.data.grade import get_verifier
from reasonops.data.models import MODEL_REGISTRY, PROMPT_TEMPLATES

from .config import (
    ATTEMPT_CAP,
    BUDGET,
    CHECK_CHARS,
    CONCURRENCY,
    HERE,
    MIN_START,
    MODELS,
    TEMPERATURE,
    TOKENIZERS,
    problems,
)

URL = "https://openrouter.ai/api/v1/chat/completions"
RUNS = HERE / "runs"
TOK_FLUSH = 800
SCORER = RULE = SEGMENT = None
TOK = {}


def load_tokenizers():
    from transformers import AutoTokenizer

    for m in MODELS:
        TOK[m] = AutoTokenizer.from_pretrained(TOKENIZERS[m], trust_remote_code=True)
    print(f"tokenizers loaded: {list(TOK)}", flush=True)


class TokenCounter:
    """Count streamed text in chunks; store provider counts separately for comparison."""

    def __init__(self, model):
        self.enc = TOK[model]
        self.n = 0
        self.buf = ""

    def add(self, text):
        self.buf += text
        if len(self.buf) >= TOK_FLUSH:
            self.flush()
        return self.n

    def flush(self):
        if self.buf:
            self.n += len(self.enc.encode(self.buf, add_special_tokens=False))
            self.buf = ""
        return self.n


def payload_for(model, prompt, cap):
    cfg = MODEL_REGISTRY[model]
    rf = cfg["reasoning_format"]
    reasoning = (
        {"effort": "high"} if rf == "effort" else {"max_tokens": cap} if rf == "max_tokens" else {}
    )
    return {
        "model": cfg["or_model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap,
        "temperature": TEMPERATURE,
        "reasoning": reasoning,
        "stream": True,
        "usage": {"include": True},
    }


async def stream_attempt(session, key, model, prompt, cap, guarded):
    reasoning, content, usage_tok = ("", "", None)
    last_check, below, abort_at, checks, capped = (0, 0, None, [], False)
    ctr = TokenCounter(model)
    async with session.stream(
        "POST",
        URL,
        json=payload_for(model, prompt, cap),
        timeout=1800,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            if d.get("usage"):
                usage_tok = d["usage"].get("completion_tokens")
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            rd, cd = (delta.get("reasoning") or "", delta.get("content") or "")
            reasoning += rd
            content += cd
            if ctr.add(rd + cd) >= cap:
                capped = True
                break
            if guarded and len(reasoning) - last_check >= CHECK_CHARS:
                last_check = len(reasoning)
                if last_check >= RULE["min"]:
                    s = SCORER.score(SEGMENT(reasoning))
                    checks.append([last_check, round(s, 4)])
                    below = below + 1 if s < RULE["tau"] else 0
                    if below >= RULE["consec"]:
                        abort_at = last_check
                        break
    counted = ctr.flush()
    complete = abort_at is None and (not capped)
    return {
        "reasoning": reasoning,
        "response": content,
        "tokens": counted,
        "usage_tokens": usage_tok,
        "aborted": abort_at is not None,
        "abort_at": abort_at,
        "capped": capped,
        "complete": complete,
        "checks": checks,
    }


async def run_one(session, sem, key, arm, model, ds, prob, verify, tmpl):
    out = RUNS / arm / model / ds / f"{str(prob['problem_id']).replace('/', '_')}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = tmpl.format(problem=prob["problem"])
    attempts, used, t0 = ([], 0, time.time())
    while BUDGET - used >= MIN_START:
        cap = min(ATTEMPT_CAP, BUDGET - used)
        async with sem:
            for retry in range(4):
                try:
                    a = await stream_attempt(session, key, model, prompt, cap, guarded=arm == "ost")
                    break
                except Exception as e:
                    a = {
                        "error": str(e),
                        "tokens": 0,
                        "aborted": False,
                        "complete": False,
                        "reasoning": "",
                        "response": "",
                        "abort_at": None,
                        "capped": False,
                        "usage_tokens": None,
                        "checks": [],
                    }
                    if retry < 3:
                        await asyncio.sleep(2**retry)
        if a["tokens"] == 0:
            break
        text = a["response"] or a["reasoning"]
        a["correct"] = bool(verify(text, prob)) if text and (not a["aborted"]) else False
        a["idx"] = len(attempts)
        used += a["tokens"]
        attempts.append(a)
    out.write_text(
        json.dumps(
            {
                "arm": arm,
                "model": model,
                "dataset": ds,
                "problem_id": prob["problem_id"],
                "budget": BUDGET,
                "tokens_used": used,
                "n_attempts": len(attempts),
                "wall_s": round(time.time() - t0, 1),
                "attempts": attempts,
            },
            ensure_ascii=False,
        )
    )
    errs = [
        abs(a["tokens"] - a["usage_tokens"]) / max(a["usage_tokens"], 1)
        for a in attempts
        if a["complete"] and a["usage_tokens"]
    ]
    print(
        f"[{arm:4s} {model:20s} {ds:13s} {str(prob['problem_id']):10s}] att={len(attempts)} abort={sum((a['aborted'] for a in attempts))} tok={used} correct={sum((a['correct'] for a in attempts))}/{len(attempts)} counterr={(max(errs) if errs else 0):.1%} {time.time() - t0:.0f}s",
        flush=True,
    )


async def main(arms):
    global SCORER, RULE, SEGMENT
    load_tokenizers()
    if "ost" in arms:
        from .ost_live import OSTScorer, segment

        SEGMENT = segment
        SCORER = OSTScorer(str(HERE / "ost_scorer_holdout.pt"), device="cpu")
        RULE = json.loads((HERE / "calibration.json").read_text())["chosen"]
        print(f"abort rule: {RULE}", flush=True)
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 16)
    async with httpx.AsyncClient(limits=limits) as session:
        tasks = []
        for ds, ps in problems().items():
            verify, tmpl = (get_verifier(ds), PROMPT_TEMPLATES[ds])
            for arm in arms:
                for m in MODELS:
                    for p in ps:
                        tasks.append(run_one(session, sem, key, arm, m, ds, p, verify, tmpl))
        print(f"{len(tasks)} runs ({arms}), budget {BUDGET} tok each", flush=True)
        await asyncio.gather(*tasks)
    print("done", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["base", "ost", "both"])
    mode = parser.parse_args().mode
    asyncio.run(main(["base", "ost"] if mode == "both" else [mode]))
