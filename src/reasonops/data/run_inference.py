#!/usr/bin/env python3
"""
Multi-model inference across benchmarks.

Backends:
  openrouter  — OpenRouter API (all non-Claude models)
  anthropic   — Anthropic API (Claude models, with extended thinking enabled)
  hf          — HuggingFace Transformers (self-hosted)

Output layout:
  {output_dir}/{model_tag}/{dataset}/{problem_id}.json
  {output_dir}/{model_tag}/{dataset}/_summary.json

Resume: already-completed problems (have 'response' field) are skipped.

Usage:
    # Single model + dataset
    python -m reasonops.data.run_inference \\
        --model qwq-32b --dataset math --output_dir /scratch/.../traces

    # All models, all datasets
    python -m reasonops.data.run_inference --all --output_dir /scratch/.../traces

    # Parallel workers (default 8)
    python -m reasonops.data.run_inference \\
        --model qwq-32b --dataset aime --workers 4 --output_dir out/
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


from reasonops.data.models import MODEL_REGISTRY, PROMPT_TEMPLATES, ALL_MODELS, ALL_DATASETS
from reasonops.data.benchmarks import load_dataset_problems
from reasonops.data.grade import get_verifier


# ── API key loading ───────────────────────────────────────────────────────────

def _load_env() -> dict:
    env_file = Path(__file__).parent.parent.parent / '.env'
    env = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env


_ENV_CACHE: dict | None = None

def _get_key(name: str) -> str:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env()
    key = os.getenv(name) or _ENV_CACHE.get(name, '')
    if not key:
        sys.exit(f'ERROR: {name} not set. Add to .env or export env var.')
    return key


# ── Async API call ────────────────────────────────────────────────────────────

_BACKEND_URLS = {
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'fireworks':  'https://api.fireworks.ai/inference/v1/chat/completions',
    'anthropic':  'https://api.anthropic.com/v1/messages',
}

_BACKEND_KEY_NAMES = {
    'openrouter': 'OPENROUTER_API_KEY',
    'fireworks':  'FIREWORKS_API_KEY',
    'anthropic':  'ANTHROPIC_API_KEY',
}



async def _api_call(
    session,
    backend: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    reasoning_format: str = 'default',
    retries: int = 8,
) -> str:
    url = _BACKEND_URLS[backend]
    key = _get_key(_BACKEND_KEY_NAMES[backend])
    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type':  'application/json',
    }
    if reasoning_format == 'effort':
        reasoning_param = {'effort': 'high'}
    elif reasoning_format == 'max_tokens':
        reasoning_param = {'max_tokens': min(max_tokens, 80000)}
    else:
        reasoning_param = {}

    payload = {
        'model':       model_id,
        'messages':    [{'role': 'user', 'content': prompt}],
        'max_tokens':  max_tokens,
        'temperature': temperature,
        'reasoning':   reasoning_param,
    }

    async with semaphore:
        for attempt in range(retries):
            try:
                if backend == 'anthropic':
                    # Claude API — extended thinking
                    # Specs: max_output 64K (Sonnet/Haiku), budget_tokens < max_tokens
                    # No temperature restriction. Header: anthropic-version 2023-06-01
                    key = _get_key('ANTHROPIC_API_KEY')
                    headers = {
                        'x-api-key': key,
                        'anthropic-version': '2023-06-01',
                        'Content-Type': 'application/json',
                    }
                    claude_max = min(max_tokens, 64000)  # actual max output for Sonnet/Haiku
                    thinking_budget = max(1024, claude_max - 4000)  # min 1024 per API requirement
                    payload = {
                        'model': model_id,
                        'max_tokens': claude_max,
                        'temperature': 1,  # REQUIRED by Anthropic API when thinking enabled
                        'thinking': {'type': 'enabled', 'budget_tokens': thinking_budget},
                        'messages': [{'role': 'user', 'content': prompt}],
                    }
                    resp = await session.post(
                        _BACKEND_URLS['anthropic'], json=payload,
                        headers=headers, timeout=600.0)
                    resp.raise_for_status()
                    data = resp.json()
                    reasoning, content = '', ''
                    for block in data.get('content', []):
                        if block.get('type') == 'thinking':
                            reasoning += block.get('thinking', '')
                        elif block.get('type') == 'text':
                            content += block.get('text', '')
                    finish_reason = data.get('stop_reason', '')
                    truncated = finish_reason == 'max_tokens'
                    return json.dumps({
                        'reasoning': reasoning, 'content': content,
                        'finish_reason': finish_reason, 'truncated': truncated,
                    })
                else:
                    # OpenRouter / Fireworks
                    resp = await session.post(url, json=payload, headers=headers,
                                              timeout=300.0)
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data['choices'][0]
                    msg = choice['message']
                    reasoning = msg.get('reasoning') or ''
                    content   = msg.get('content') or ''
                    finish_reason = choice.get('finish_reason', '')
                    truncated = finish_reason == 'length'
                    # If model returned reasoning inline (no separate field),
                    # use content as reasoning trace
                    if not reasoning and content:
                        reasoning = content
                    return json.dumps({
                        'reasoning': reasoning, 'content': content,
                        'finish_reason': finish_reason, 'truncated': truncated,
                    })
            except Exception as e:
                is_rate_limit = '429' in str(e)
                if attempt < retries - 1:
                    if is_rate_limit:
                        wait = min(10 * (2 ** attempt), 120)  # 10s, 20s, 40s, 80s, 120s
                    else:
                        wait = 2 ** attempt
                    print(f'  [retry {attempt+1}/{retries}] {e} — waiting {wait}s',
                          file=sys.stderr, flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f'  [fail] {model_id} on prompt: {e}',
                          file=sys.stderr, flush=True)
                    return ''
    return ''


# ── HF backend ────────────────────────────────────────────────────────────────

def _load_hf_model(hf_id: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f'Loading {hf_id} via HuggingFace...')
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype='auto', device_map='auto', trust_remote_code=True)
    model.eval()
    print(f'Loaded on {next(model.parameters()).device}')
    return tok, model


def _generate_hf(tok, model, prompt: str,
                 max_new_tokens: int = 4096, temperature: float = 0.0) -> str:
    import torch
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


# ── Single dataset run ────────────────────────────────────────────────────────

async def run_dataset(
    model_tag: str,
    dataset_name: str,
    output_root: Path,
    n_problems: int | None,
    temperature: float,
    max_new_tokens_override: int | None,
    workers: int,
):
    model_cfg     = MODEL_REGISTRY[model_tag]
    backend       = model_cfg['backend']
    model_id      = model_cfg['or_model_id']
    max_tokens    = max_new_tokens_override or model_cfg['max_new_tokens']
    prompt_tmpl   = PROMPT_TEMPLATES[dataset_name]
    verifier      = get_verifier(dataset_name)

    out_dir = output_root / model_tag / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_dataset_problems(dataset_name, n=n_problems)
    print(f'\n[{model_cfg["display_name"]} × {dataset_name}]  '
          f'n={len(problems)}  backend={backend}  max_tok={max_tokens}')

    counts = {'done': 0, 'skipped': 0, 'correct': 0, 'total': len(problems)}

    def _safe_filename(problem_id: str) -> str:
        """Sanitize problem_id for use as a filename (handles MATH's path-like IDs)."""
        s = problem_id.replace('/', '_')
        if s.endswith('.json'):
            s = s[:-5]
        return s

    if backend == 'hf':
        tok, hf_model = _load_hf_model(model_id)
        for prob in problems:
            out_path = out_dir / f"{_safe_filename(prob['problem_id'])}.json"
            if _is_done(out_path):
                _count_existing(out_path, counts)
                continue
            prompt   = prompt_tmpl.format(problem=prob['problem'])
            response = _generate_hf(tok, hf_model, prompt, max_tokens, temperature)
            _save(out_path, prob, model_tag, model_id, backend, response, verifier, counts)
        return counts

    # Async API backend
    import httpx
    semaphore = asyncio.Semaphore(workers)

    async with httpx.AsyncClient() as session:
        think_prefix = model_cfg.get('think_prefix', False)

        async def process(prob):
            out_path = out_dir / f"{_safe_filename(prob['problem_id'])}.json"
            if _is_done(out_path):
                _count_existing(out_path, counts)
                return
            base_prompt = prompt_tmpl.format(problem=prob['problem'])
            prompt = f"/think\n{base_prompt}" if think_prefix else base_prompt
            response = await _api_call(
                session, backend, model_id, prompt,
                max_tokens, temperature, semaphore,
                reasoning_format=model_cfg.get('reasoning_format', 'default'))
            if not response:
                print(f'  [skip] {prob["problem_id"]} — empty API response, not saving',
                      flush=True)
                return
            _save(out_path, prob, model_tag, model_id, backend, response, verifier, counts)
            _print_progress(counts, prob['problem_id'])

        await asyncio.gather(*[process(p) for p in problems])

    return counts


def _is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
        return bool(rec.get('reasoning') or rec.get('response'))
    except Exception:
        return False


def _count_existing(path: Path, counts: dict):
    counts['skipped'] += 1
    counts['done'] += 1
    try:
        if json.loads(path.read_text()).get('correct'):
            counts['correct'] += 1
    except Exception:
        pass


def _save(out_path: Path, prob: dict, model_tag: str, model_id: str,
          backend: str, response: str, verifier, counts: dict):
    # response is JSON {"reasoning": ..., "content": ...} from API backend
    try:
        parsed   = json.loads(response) if response else {}
        reasoning = parsed.get('reasoning', '')
        content   = parsed.get('content', '')
    except Exception:
        reasoning, content = '', response or ''

    # Verify against content; fall back to full reasoning if content is empty
    verify_text = content if content else reasoning
    correct = verifier(verify_text, prob) if verify_text else None
    counts['done'] += 1
    if correct:
        counts['correct'] += 1
    record = {
        **{k: v for k, v in prob.items() if k not in ('test_code',)},
        'model_tag': model_tag,
        'model_id':  model_id,
        'backend':   backend,
        'reasoning': reasoning,   # the CoT trace — feeds into our segmentation pipeline
        'response':  content,     # final answer only
        'correct':   correct,
        'finish_reason': parsed.get('finish_reason', ''),
        'truncated':     parsed.get('truncated', False),
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False))


def _print_progress(counts: dict, problem_id: str):
    n      = counts['done']
    total  = counts['total']
    n_new  = n - counts['skipped']
    acc    = counts['correct'] / max(n_new, 1)
    print(f'  [{n}/{total}] {problem_id}  acc={acc:.2%}', flush=True)


def _write_summary(out_dir: Path, model_tag: str, model_id: str,
                   dataset: str, counts: dict):
    n_new    = counts['done'] - counts['skipped']
    accuracy = counts['correct'] / max(n_new, 1)
    summary  = {
        'model_tag':  model_tag,
        'model_id':   model_id,
        'dataset':    dataset,
        'n_problems': counts['total'],
        'n_new':      n_new,
        'n_correct':  counts['correct'],
        'accuracy':   round(accuracy, 4),
    }
    (out_dir / '_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'  Accuracy: {accuracy:.3f} ({counts["correct"]}/{n_new} new)')
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--all',     action='store_true',
                     help='Run all models × all datasets')
    src.add_argument('--model',   help=f'Model tag. One of: {ALL_MODELS}')

    ap.add_argument('--dataset',       default=None,
                    help=f'Dataset (required unless --all). One of: {ALL_DATASETS}')
    ap.add_argument('--output_dir',    required=True)
    ap.add_argument('--n_problems',    type=int, default=None,
                    help='Override sample size (default from datasets.py)')
    ap.add_argument('--temperature',   type=float, default=0.0)
    ap.add_argument('--max_new_tokens', type=int, default=None)
    ap.add_argument('--workers',       type=int, default=8,
                    help='Parallel API calls per model-dataset pair (default 8)')
    args = ap.parse_args()

    out_root = Path(args.output_dir)

    if args.all:
        pairs = [(m, d) for m in ALL_MODELS for d in ALL_DATASETS]
    else:
        if args.model not in MODEL_REGISTRY:
            ap.error(f'Unknown model {args.model!r}. Available: {ALL_MODELS}')
        if args.dataset is None:
            ap.error('--dataset required unless --all')
        if args.dataset not in ALL_DATASETS:
            ap.error(f'Unknown dataset {args.dataset!r}. Available: {ALL_DATASETS}')
        pairs = [(args.model, args.dataset)]

    all_summaries = []
    for model_tag, dataset_name in pairs:
        counts = asyncio.run(run_dataset(
            model_tag, dataset_name, out_root,
            args.n_problems, args.temperature, args.max_new_tokens, args.workers,
        ))
        out_dir = out_root / model_tag / dataset_name
        cfg = MODEL_REGISTRY[model_tag]
        summary = _write_summary(out_dir, model_tag, cfg['or_model_id'],
                                 dataset_name, counts)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        (out_root / '_all_summaries.json').write_text(
            json.dumps(all_summaries, indent=2))
        print(f'\nAll summaries → {out_root / "_all_summaries.json"}')


if __name__ == '__main__':
    main()
