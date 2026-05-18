#!/usr/bin/env python3
"""
LLM-based grading of reasoning traces.

For non-code datasets: Claude judges whether the model's answer matches gold.
For code datasets (humaneval, livecodebench): LLM extracts code → execution.

Usage:
    python -m reasonops.data.grade \\
        --input   /scratch/.../traces.jsonl.gz \\
        --output  /scratch/.../traces_graded.jsonl.gz \\
        --report  /scratch/.../grade_report.json \\
        --workers 8

Checkpointing: re-run with the same --output path to resume.
Traces already labeled (correct is not None) are skipped.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic


# ═════════════════════════════════════════════════════════════════════════════
# Answer verification utilities (used by both grade.py and run_inference.py)
# ═════════════════════════════════════════════════════════════════════════════

_HAS_MATH_VERIFY = False
try:
    from math_verify.parser import parse as mv_parse, LatexExtractionConfig, ExprExtractionConfig
    from math_verify.grader import verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    print("WARNING: math-verify not installed — regex fallback for math grading. "
          "pip install math-verify")


def get_verifier(dataset: str):
    """Return verifier(response: str, prob: dict) -> bool | None"""
    verifiers = {
        "math":           _verify_math,
        "gpqa":           _verify_mcq,
        "aime":           _verify_aime,
        "livecodebench":  _verify_code,
        "hle":            _verify_hle,
        "mmlu_pro":       _verify_mcq,
        "arc_challenge":  _verify_mcq,
        "bbh":            _verify_bbh,
        "humaneval":      _verify_code,
        "arc_agi2":       _verify_arc_agi2,
    }
    if dataset not in verifiers:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return verifiers[dataset]


def _extract_boxed(text: str) -> str | None:
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def _extract_answer_field(text: str) -> str | None:
    m = re.search(r'(?i)Answer\s*:\s*([^\n]+)', text)
    return m.group(1).strip() if m else None


def _verify_math_sympy(predicted: str, gold: str) -> bool | None:
    if not _HAS_MATH_VERIFY:
        return None
    try:
        gold_parsed = mv_parse(gold, extraction_config=(LatexExtractionConfig(), ExprExtractionConfig()))
        pred_parsed = mv_parse(predicted, extraction_config=(LatexExtractionConfig(), ExprExtractionConfig()))
        if not gold_parsed or not pred_parsed:
            return None
        return mv_verify(gold_parsed, pred_parsed)
    except Exception:
        return None


def _norm_math_fallback(s: str) -> str:
    s = s.strip().rstrip('.')
    s = re.sub(r'\s+', '', s)
    s = s.replace('$', '').replace(',', '').replace('%', '')
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    s = re.sub(r'\\[dt]frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\(?:text|mathrm|mathbf|textbf|left|right|big|Big)\s*', '', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace('{', '').replace('}', '')
    return s.lower()


def _try_numeric_equal(a: str, b: str) -> bool | None:
    try:
        fa = float(a.replace(' ', ''))
        fb = float(b.replace(' ', ''))
        return abs(fa - fb) < 1e-6
    except (ValueError, OverflowError):
        return None


def _verify_math(response: str, prob: dict) -> bool | None:
    gt = str(prob['answer'])
    predicted = _extract_boxed(response)
    if predicted is None:
        predicted = _extract_answer_field(response)
    if predicted is None:
        nums = re.findall(r'-?\d+(?:[.,]\d+)*(?:/\d+)?', response)
        predicted = nums[-1] if nums else None
    if predicted is None:
        return None
    result = _verify_math_sympy(predicted, gt)
    if result is not None:
        return result
    if _norm_math_fallback(predicted) == _norm_math_fallback(gt):
        return True
    numeric = _try_numeric_equal(_norm_math_fallback(predicted), _norm_math_fallback(gt))
    return numeric if numeric is not None else False


def _verify_aime(response: str, prob: dict) -> bool | None:
    gt = int(prob['answer'])
    boxed = _extract_boxed(response)
    if boxed is not None:
        nums = re.findall(r'\d+', boxed)
        if nums:
            val = int(nums[0])
            if 0 <= val <= 999:
                return val == gt
    answer_field = _extract_answer_field(response)
    if answer_field:
        nums = re.findall(r'\d+', answer_field)
        if nums:
            val = int(nums[0])
            if 0 <= val <= 999:
                return val == gt
    m = re.search(r'the answer is\s+(\d{1,3})\b', response, re.IGNORECASE)
    if m:
        return int(m.group(1)) == gt
    nums = re.findall(r'\b(\d{1,3})\b', response)
    if nums:
        val = int(nums[-1])
        if 0 <= val <= 999:
            return val == gt
    return None


def _extract_mcq_answer(response: str, valid: str = 'ABCDEFGHIJ') -> str | None:
    m = re.search(rf'(?:The answer is|the answer is|The answer:)\s*\(?([{valid}])\)?',
                  response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(rf'(?i)Answer\s*:\s*\$?\(?([{valid}])\)?\$?', response)
    if m:
        return m.group(1).upper()
    m = re.search(rf'(?:choose|pick|select|Choice)\s*:?\s*\(?([{valid}])\)?',
                  response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(rf'\(([{valid}])\)', response)
    if matches:
        return matches[-1].upper()
    m = re.search(rf'\*\*\(?([{valid}])\)?\*\*', response)
    if m:
        return m.group(1).upper()
    matches = re.findall(rf'\b([{valid}])\b', response)
    if matches:
        return matches[-1].upper()
    return None


def _verify_mcq(response: str, prob: dict) -> bool | None:
    predicted = _extract_mcq_answer(response)
    if predicted is None:
        return None
    gt = prob['answer'].strip().upper().strip('()')
    return predicted == gt


def _extract_code(response: str) -> str | None:
    m = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = response.split('\n')
    code_lines = []
    capture = False
    for line in lines:
        if re.match(r'^(def |class |import |from |if |for |while |    )', line):
            capture = True
        if capture:
            code_lines.append(line)
        elif code_lines and line.strip() == '':
            code_lines.append(line)
        elif code_lines:
            break
    return '\n'.join(code_lines).strip() if code_lines else None


def _run_code_safely(code: str, stdin: str = '', timeout: int = 10) -> tuple[bool, str]:
    fname = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, encoding='utf-8') as f:
            f.write(code)
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            input=stdin, timeout=timeout,
            capture_output=True, text=True,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, ''
    except Exception:
        return False, ''
    finally:
        if fname:
            Path(fname).unlink(missing_ok=True)


def _verify_code(response: str, prob: dict) -> bool | None:
    code = _extract_code(response)
    if code is None:
        return None
    test_code   = prob.get('test', prob.get('test_code', ''))
    entry_point = prob.get('entry_point', '')
    if test_code and entry_point:
        success, _ = _run_code_safely(f"{code}\n\n{test_code}\n\ncheck({entry_point})\n")
        return success
    test_cases = prob.get('test_cases', [])
    if test_cases:
        tc = test_cases[0] if isinstance(test_cases, list) else {}
        if isinstance(tc, str):
            tc = json.loads(tc)
        inp      = tc.get('input', '')
        expected = tc.get('output', '').strip()
        success, stdout = _run_code_safely(code, stdin=inp)
        return stdout == expected if success else False
    return None


def _verify_hle(response: str, prob: dict) -> bool | None:
    answer_type = prob.get('answer_type', 'exact_match')
    gt = str(prob['answer']).strip()
    if answer_type == 'multiple_choice' or re.match(r'^[A-Z]$', gt):
        return _verify_mcq(response, prob)
    predicted = _extract_answer_field(response)
    if predicted is None:
        predicted = response.strip().split('\n')[-1].strip()
    result = _verify_math_sympy(predicted, gt)
    if result is not None:
        return result
    if _norm_math_fallback(predicted) == _norm_math_fallback(gt):
        return True
    if len(gt) > 2 and gt.lower() in predicted.lower():
        return True
    return False


def _verify_bbh(response: str, prob: dict) -> bool | None:
    gt = str(prob['answer']).strip()
    m = re.search(
        r'(?:The answer is|the answer is|The answer:|The final answer:)\s*(.+?)(?:\.|$)',
        response, re.IGNORECASE)
    predicted = m.group(1).strip() if m else _extract_answer_field(response)
    if predicted is None:
        return None
    predicted = predicted.strip().strip('()').strip()
    if re.match(r'^\(?[A-Z]\)?$', gt):
        gt_letter   = gt.strip('()')
        pred_letter = predicted.strip('()').upper()
        if len(pred_letter) == 1 and pred_letter.isalpha():
            return pred_letter == gt_letter
    if gt.lower() in ('yes', 'no', 'true', 'false', 'valid', 'invalid'):
        return predicted.lower() == gt.lower()
    return _norm_math_fallback(predicted) == _norm_math_fallback(gt)


def _verify_arc_agi2(response: str, prob: dict) -> bool | None:
    expected = prob.get('answer')
    if expected is None:
        return None
    m = re.search(r'(\[\s*\[[\d\s,\[\]]+\]\s*\])', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)) == expected
        except json.JSONDecodeError:
            pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# LLM judge
# ═════════════════════════════════════════════════════════════════════════════

MODEL_FAST   = "claude-haiku-4-5-20251001"
MODEL_STRONG = "claude-sonnet-4-6"
CODE_DATASETS = {"humaneval", "livecodebench"}
SKIP_DATASETS = {"arc_agi2"}
RESP_TAIL     = 2000

import threading
_tlocal = threading.local()

def _get_client() -> anthropic.Anthropic:
    if not hasattr(_tlocal, "client"):
        _tlocal.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _tlocal.client


def _call_llm(prompt: str, max_tokens: int = 32, retries: int = 4,
              strong: bool = False) -> str:
    model = MODEL_STRONG if strong else MODEL_FAST
    client = _get_client()
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            time.sleep(min(60, 4 ** attempt))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return "UNCLEAR"


def _tail(text: str, n: int = RESP_TAIL) -> str:
    return text[-n:] if len(text) > n else text


def judge_llm(response: str, prob: dict) -> tuple:
    gold    = str(prob.get("answer", "")).strip()
    problem = str(prob.get("problem", ""))[:1200]
    prompt  = (
        f"You are a strict but fair grader. Determine if the student's final answer "
        f"matches the correct answer.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct answer: {gold}\n\n"
        f"Student's response (final portion):\n{_tail(response, 1500)}\n\n"
        f"Does the student's FINAL answer match the correct answer? "
        f"Focus only on their final stated answer, not their work. "
        f"Equivalent expressions (e.g. 1/2 = 0.5, 'option I' = choice I) count as correct. "
        f"Reply with only YES or NO."
    )
    raw = _call_llm(prompt, max_tokens=8, strong=True).upper().strip()
    if raw.startswith("YES"):
        return True, "llm_judge"
    if raw.startswith("NO"):
        return False, "llm_judge"
    return None, "llm_unclear"


def _extract_code_llm(response: str, prob: dict, dataset: str) -> str | None:
    problem = str(prob.get("problem", ""))[:1000]
    if dataset == "humaneval":
        entry  = prob.get("entry_point", "solution")
        prompt = (
            f"Extract ONLY the Python function `{entry}` implementation from the student's response. "
            f"Include any helper functions it calls. "
            f"Do NOT include test code, example usage, explanations, or markdown fences. "
            f"Output raw Python only.\n\nProblem:\n{problem}\n\nStudent response:\n{response}"
        )
    else:
        prompt = (
            f"Extract ONLY the complete Python solution from the student's response. "
            f"The program reads from stdin and prints to stdout. "
            f"Include all necessary imports. "
            f"Do NOT include explanations or markdown fences. "
            f"Output raw Python only.\n\nProblem:\n{problem}\n\nStudent response:\n{response}"
        )
    raw = _call_llm(prompt, max_tokens=2048, strong=True)
    raw = re.sub(r"^```(?:python)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw.strip())
    return raw.strip() if raw.strip() else None


def judge_code(response: str, prob: dict, dataset: str) -> tuple:
    code = _extract_code_llm(response, prob, dataset)
    if not code:
        return None, "llm_no_code"

    if dataset == "humaneval":
        test_code = prob.get("test", prob.get("test_code", ""))
        entry     = prob.get("entry_point", "")
        if not test_code or not entry:
            return None, "llm_missing_harness"
        success, _ = _run_code_safely(f"{code}\n\n{test_code}\n\ncheck({entry})\n")
        return success, "llm_extract_exec_humaneval"
    else:
        test_cases = prob.get("test_cases", [])
        if not test_cases:
            return None, "llm_missing_testcases"
        for tc in test_cases:
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except json.JSONDecodeError:
                    continue
            inp      = tc.get("input", "")
            expected = tc.get("output", "").strip()
            ok, stdout = _run_code_safely(code, stdin=inp)
            if not ok:
                return False, "llm_extract_exec_livecodebench"
            if stdout != expected:
                return False, "llm_extract_exec_livecodebench"
        return True, "llm_extract_exec_livecodebench"


def _grade_trace(trace: dict) -> tuple:
    ds       = trace.get("dataset", "")
    response = trace.get("reasoning") or trace.get("model_response") or ""
    prob     = {
        "problem":     trace.get("problem", ""),
        "answer":      trace.get("gold_answer", ""),
        "test":        trace.get("test", ""),
        "test_code":   trace.get("test_code", ""),
        "entry_point": trace.get("entry_point", ""),
        "test_cases":  trace.get("test_cases", []),
    }
    if ds in CODE_DATASETS:
        return judge_code(response, prob, ds)
    return judge_llm(response, prob)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--report",     required=True)
    ap.add_argument("--benchmarks", default=None,
                    help="benchmark_lookup.json (needed for humaneval/livecodebench harness fields)")
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--dry_run",    action="store_true")
    ap.add_argument("--dataset",    default=None, help="Only grade this dataset")
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--grade_model_fast",   default="claude-haiku-4-5-20251001",
                    help="Model used for fast/cheap grading calls")
    ap.add_argument("--grade_model_strong", default="claude-sonnet-4-6",
                    help="Model used for strong/judge grading calls")
    args = ap.parse_args()

    global MODEL_FAST, MODEL_STRONG
    MODEL_FAST   = args.grade_model_fast
    MODEL_STRONG = args.grade_model_strong

    benchmark_lookup: dict = {}
    if args.benchmarks:
        bl_path = Path(args.benchmarks)
        if not bl_path.exists():
            raise SystemExit(f"--benchmarks not found: {bl_path}")
        benchmark_lookup = json.loads(bl_path.read_text())
        print(f"Loaded benchmark lookup: {len(benchmark_lookup):,} entries")
    else:
        print("WARNING: no --benchmarks; humaneval/livecodebench will be skipped")

    opener = gzip.open if args.input.endswith(".gz") else open
    traces = []
    with opener(args.input, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            ds  = t.get("dataset", "")
            pid = t.get("problem_id", "")
            for key in (f"{ds}|{pid}", f"{ds}|{pid.replace('/', '_')}"):
                rec = benchmark_lookup.get(key)
                if rec:
                    for field, val in rec.items():
                        if field not in t or t[field] is None:
                            t[field] = val
                    break
            traces.append(t)
    print(f"Loaded {len(traces):,} traces")

    labeled: dict[str, tuple] = {}
    out_path = Path(args.output)
    if out_path.exists():
        opener_out = gzip.open if args.output.endswith(".gz") else open
        with opener_out(args.output, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                if t.get("correct") is not None:
                    labeled[t["trace_id"]] = (t["correct"], t.get("correct_method", ""))
        print(f"Resuming: {len(labeled):,} already labeled")

    effective_skip = set(SKIP_DATASETS)
    if not benchmark_lookup:
        effective_skip |= CODE_DATASETS

    candidates = [
        t for t in traces
        if not t.get("truncated")
        and t.get("dataset") not in effective_skip
        and t["trace_id"] not in labeled
        and (args.dataset is None or t.get("dataset") == args.dataset)
    ]
    if args.limit:
        candidates = candidates[:args.limit]

    by_ds = Counter(t["dataset"] for t in candidates)
    print(f"Candidates: {len(candidates):,}")
    for ds, n in sorted(by_ds.items()):
        print(f"  {ds}: {n:,}")

    if args.dry_run:
        print("Dry run — exiting")
        return

    stats  = defaultdict(Counter)
    errors = []

    def process(t: dict):
        try:
            label, method = _grade_trace(t)
            return t["trace_id"], label, method, t.get("dataset", "?"), None
        except Exception as e:
            return t["trace_id"], None, "error", t.get("dataset", "?"), str(e)

    done = 0
    id_to_result: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, t): t for t in candidates}
        for fut in as_completed(futures):
            trace_id, label, method, ds, err = fut.result()
            id_to_result[trace_id] = (label, method)
            stats[ds]["total"] += 1
            if err:
                stats[ds]["error"] += 1
                errors.append({"trace_id": trace_id, "error": err})
            elif label is None:
                stats[ds]["unclear"] += 1
            elif label:
                stats[ds]["true"] += 1
            else:
                stats[ds]["false"] += 1
            done += 1
            if done % 200 == 0:
                print(f"  [{done:,}/{len(candidates):,}]...", flush=True)

    labeled.update(id_to_result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener_out = gzip.open if args.output.endswith(".gz") else open
    with opener_out(args.output, "wt") as fo:
        for t in traces:
            tid = t["trace_id"]
            if tid in labeled:
                lbl, meth = labeled[tid]
                t["correct"]        = lbl
                t["correct_method"] = meth
            fo.write(json.dumps(t) + "\n")

    print(f"\n=== Grading Report ===")
    for ds in sorted(stats):
        s = stats[ds]
        pct = s["true"] / max(s["total"] - s["unclear"] - s["error"], 1)
        print(f"  {ds:<18} total={s['total']:,}  correct={s['true']:,} ({pct:.1%})  "
              f"unclear={s['unclear']:,}  errors={s['error']:,}")

    report = {
        "n_input":    len(traces),
        "n_labeled":  len(labeled),
        "per_dataset": {ds: dict(s) for ds, s in stats.items()},
        "errors":     errors[:50],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote: {args.output}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
