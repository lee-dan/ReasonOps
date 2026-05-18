#!/usr/bin/env python3
"""
Operator discovery pipeline.

Extracts discourse pivot phrases from reasoning traces, embeds them with
e5-small-v2, and clusters into K operators via KMeans. Each span in the
corpus is then labeled with its pivot's cluster assignment.

Steps:
  1. Sentence segmentation and pivot extraction (first 3 alpha tokens).
  2. Frequency + domain filter: keep pivots in ≥100 traces across ≥3 datasets
     whose tokens all appear in the top-N corpus vocabulary.
  3. e5-small-v2 embedding (384-dim, L2-normalized) of each pivot phrase.
  4. KMeans clustering; K selected externally by LLM-judge κ sweep.
  5. LLM-based operator naming per cluster.
  6. Span relabeling: each trace span gets its pivot's cluster id.

Usage:
    python -m reasonops.operators.discover_operators \\
        --input       /scratch/.../traces_final.jsonl.gz \\
        --output_dir  /scratch/.../operators/discovered_k7 \\
        --n_operators 7
"""

import argparse, gzip, json, math, re, sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

KEEP_MODELS = {
    "claude-haiku", "claude-sonnet",
    "grok-3-mini", "kimi-k2", "kimi-k2.5",
    "nemotron-49b", "qwen3-235b-thinking", "qwen3-30b-thinking",
    "qwq-32b", "r1-0528", "r1-distill-llama-70b", "r1-distill-qwen-32b",
}

_BULLET_RE   = re.compile(r'^\s*(?:\d+[\.\)]\s+|[-•*]\s+)')
_CODE_BLOCK  = re.compile(r'```[\s\S]*?```|`[^`\n]+`')
_MATH_INLINE = re.compile(r'\$[^\$\n]{1,80}\$|\\\([^\)]{1,80}\\\)')



def split_sentences(text: str) -> list[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _CODE_BLOCK.sub(' [CODE] ', text)
    text = _MATH_INLINE.sub(' [MATH] ', text)
    raw = []
    for para in re.split(r'\n{2,}', text):
        para = para.strip()
        if not para:
            continue
        lines = para.split('\n')
        if len(lines) == 1:
            raw.extend(_split_inline(lines[0]))
        else:
            for line in lines:
                line = line.strip()
                if line:
                    raw.extend(_split_inline(line) if len(line) >= 120 else [line])
    return [s.strip() for s in raw
            if s.strip() and len(s.strip()) >= 4
            and s.strip() not in ('[CODE]', '[MATH]')]


def _split_inline(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\["])', text)
    return [p.strip() for p in parts if p.strip()]


def alpha_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def sentence_start(text: str, n: int = 3) -> tuple[str, ...]:
    if not text.lstrip()[:1].isupper():
        return ()
    return tuple(alpha_tokens(text)[:n])


def is_structural_pivot(sent: str, prev: str | None, idx: int) -> bool:
    if idx == 0:
        return True
    if _BULLET_RE.match(sent):
        return True
    if sent.strip().endswith('?') and prev and not prev.strip().endswith('?'):
        return True
    return False



def build_top_vocab(trace_sentences: dict, top_n: int = 2000) -> frozenset:
    freq = Counter()
    for sents in trace_sentences.values():
        for s in sents:
            freq.update(alpha_tokens(s))
    top = {tok for tok, _ in freq.most_common(top_n)}
    print(f'  Top-{top_n} vocab built from {sum(freq.values()):,} tokens, '
          f'{len(freq):,} unique')
    return frozenset(top)



def find_type_a_starts(trace_sentences: dict, min_trace_freq: int,
                        top_vocab: frozenset,
                        trace_meta: dict = None,
                        min_domain_count: int = 1,
                        top_k: int = 50_000,
                        n_start_tokens: int = 3) -> frozenset:
    """
    Type-A pivots: appear in >= min_trace_freq traces, >= min_domain_count
    datasets, AND all tokens are in top_vocab (domain-immunity filter).
    """
    start_freq   = Counter()
    start_traces = defaultdict(set)
    start_domains = defaultdict(set)

    for tid, sents in trace_sentences.items():
        domain = trace_meta[tid]['dataset'] if trace_meta else 'unk'
        for s in sents:
            st = sentence_start(s, n=n_start_tokens)
            if st:
                start_freq[st] += 1
                start_traces[st].add(tid)
                start_domains[st].add(domain)

    print(f'  Unique sentence starts: {len(start_freq):,}')

    top = []
    for st, _ in start_freq.most_common(top_k):
        if len(start_traces[st]) < min_trace_freq:
            continue
        if min_domain_count > 1 and len(start_domains[st]) < min_domain_count:
            continue
        if not all(t in top_vocab for t in st):
            continue
        top.append(st)

    print(f'  After filters (>={min_trace_freq} traces, >={min_domain_count} domains, '
          f'top-{len(top_vocab)} vocab): {len(top):,} type-A starts')
    return frozenset(top)



def segment_traces(trace_sentences: dict, type_a_starts: frozenset,
                   max_body: int = 50, n_start_tokens: int = 3) -> dict:
    from tqdm import tqdm
    result = {}
    for tid, sents in tqdm(trace_sentences.items(), desc='Segmenting', unit='trace'):
        spans    = []
        anchor   = None
        body     = []
        ngram    = ''
        span_idx = 0

        for i, sent in enumerate(sents):
            prev      = sents[i - 1] if i > 0 else None
            st        = sentence_start(sent, n=n_start_tokens)
            is_type_a = st in type_a_starts
            is_str    = is_structural_pivot(sent, prev, i)
            is_piv    = is_type_a or is_str

            if is_piv:
                if anchor is not None:
                    spans.append({
                        'anchor':        anchor,
                        'body':          body[:max_body],
                        'body_len':      len(body),
                        'span_idx':      span_idx,
                        'anchor_reason': 'type_a' if ngram else 'structural',
                        'ngram':         ngram,
                        'cluster':       -1,
                    })
                    span_idx += 1
                anchor = sent
                body   = []
                ngram  = ' '.join(st) if is_type_a else ''
            else:
                if anchor is None:
                    anchor = sent
                    ngram  = ''
                else:
                    body.append(sent)

        if anchor is not None:
            spans.append({
                'anchor':        anchor,
                'body':          body[:max_body],
                'body_len':      len(body),
                'span_idx':      span_idx,
                'anchor_reason': 'type_a' if ngram else 'structural',
                'ngram':         ngram,
                'cluster':       -1,
            })

        result[tid] = spans
    return result



def load_e5(model_name: str = 'intfloat/e5-small-v2'):
    from sentence_transformers import SentenceTransformer
    print(f'  Loading {model_name}...')
    model = SentenceTransformer(model_name)
    print(f'  Embedding dim: {model.get_sentence_embedding_dimension()}')
    return model


def embed_pivots(starts_list: list, e5_model, batch_size: int = 512) -> np.ndarray:
    phrases = [' '.join(st) for st in starts_list]
    vecs = e5_model.encode(phrases, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
    return vecs.astype(np.float64)



def cluster_type_a_starts(type_a_starts: frozenset, e5_model,
                           n_operators: int, seed: int = 42):
    from sklearn.cluster import KMeans

    starts_list = sorted(type_a_starts)
    print(f'  Embedding {len(starts_list):,} type-A pivots with e5-small...')
    X = embed_pivots(starts_list, e5_model)

    print(f'  KMeans k={n_operators} on {X.shape}...')
    km = KMeans(n_clusters=n_operators, random_state=seed,
                n_init=30, max_iter=500)
    labels = km.fit_predict(X)

    start_to_cluster = {st: int(lab)
                        for st, lab in zip(starts_list, labels)}

    counts = Counter(labels.tolist())
    print(f'  Cluster sizes (by pivot count): '
          f'{sorted(counts.values(), reverse=True)}')

    return start_to_cluster


def assign_span_clusters(trace_spans: dict, start_to_cluster: dict) -> None:
    """Assign clusters to type-A spans via dict lookup. Structural spans stay -1."""
    from tqdm import tqdm
    n_type_a = n_struct = 0
    all_spans = [(sp, tid) for tid, spans in trace_spans.items() for sp in spans]
    for sp, _ in tqdm(all_spans, desc='Assigning clusters', unit='span'):
        st = sentence_start(sp['anchor'])
        if st in start_to_cluster:
            sp['cluster'] = start_to_cluster[st]
            n_type_a += 1
        else:
            sp['cluster'] = -1
            n_struct += 1
    print(f'  Assigned: {n_type_a:,} type-A  {n_struct:,} structural (left as -1)')



def compute_op_stats(trace_spans: dict, all_domains: list) -> dict:
    n_domains  = len(all_domains)
    op_counts  = defaultdict(int)
    op_domain  = defaultdict(Counter)
    op_ngrams  = defaultdict(Counter)

    for spans in trace_spans.values():
        for sp in spans:
            cid = sp['cluster']
            if cid < 0:
                continue
            op_counts[cid] += 1
            op_domain[cid][sp.get('dataset', 'unk')] += 1
            ng = sp.get('ngram', '').strip()
            if ng:
                op_ngrams[cid][ng] += 1

    op_stats = {}
    for op_id in sorted(op_counts):
        dom = op_domain[op_id]
        op_stats[op_id] = {
            'n_sentences':   op_counts[op_id],
            'top_starts':    [s for s, _ in op_ngrams[op_id].most_common(25)],
            'domain_counts': dict(dom.most_common()),
            'entropy':       round(_norm_entropy(Counter(dom), n_domains), 3),
        }
    return op_stats


def _norm_entropy(counts: Counter, n_total: int) -> float:
    total = sum(counts.values())
    if total == 0 or n_total < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return h / math.log(n_total)



_LLM_NAME_PROMPT = """\
You are an expert in cognitive science and discourse analysis.

Below are {k} clusters discovered from LLM chain-of-thought reasoning traces.
Each cluster groups spans by their anchor pivot phrase. For each cluster:
  - Top anchor phrases (the surface signal that defines the cluster)
  - 5 full span examples (anchor + body)

Name each cluster based on the COGNITIVE/EPISTEMIC ACT being performed.

Rules:
- SHORT (1-2 words, ALL_CAPS): EXPLORING, REVISING, CONCLUDING, PLANNING, etc.
- Name must reflect the COGNITIVE ACT, not topical content
- Mutually exclusive and collectively exhaustive

Clusters:
{cluster_block}

Respond EXACTLY:

CLUSTER: <id>
NAME: OPERATOR_NAME
DESCRIPTION: one sentence on the cognitive move
---
"""


def name_clusters_llm(op_stats: dict, trace_spans: dict,
                       model: str = 'claude-sonnet-4-6') -> dict:
    import os, time
    try:
        import anthropic as _ant
    except ImportError:
        sys.exit('ERROR: anthropic package required')

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        sys.exit('ERROR: ANTHROPIC_API_KEY not set')

    exemplars = defaultdict(list)
    for tid, spans in trace_spans.items():
        for sp in spans:
            cid = sp['cluster']
            if cid >= 0 and len(exemplars[cid]) < 5:
                anchor    = sp['anchor'].replace('\n', ' ')[:120]
                body_text = ' '.join(s.replace('\n', ' ')[:80]
                                     for s in sp.get('body', [])[:4])
                full      = anchor + (' ' + body_text if body_text else '')
                exemplars[cid].append(full[:400])

    lines = []
    for op_id in sorted(op_stats):
        s      = op_stats[op_id]
        starts = s['top_starts'][:12]
        exs    = exemplars[op_id][:5]
        lines.append(f'Cluster {op_id}  (n={s["n_sentences"]:,}  entropy={s["entropy"]:.2f})')
        lines.append(f'  Top anchor phrases: {", ".join(repr(x) for x in starts)}')
        lines.append(f'  Examples:')
        for i, ex in enumerate(exs, 1):
            lines.append(f'    [{i}] "{ex}"')
        lines.append('')

    prompt = _LLM_NAME_PROMPT.format(
        k=len(op_stats), cluster_block='\n'.join(lines))

    print(f'  Calling {model} to name {len(op_stats)} clusters...')
    client = _ant.Anthropic(api_key=api_key)
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=model, max_tokens=2000, temperature=0.0,
                messages=[{'role': 'user', 'content': prompt}],
            )
            response = msg.content[0].text.strip()
            break
        except Exception as e:
            print(f'  Error (attempt {attempt+1}): {e}')
            time.sleep(2 ** (attempt + 1))
    else:
        return {op_id: {'name': f'OPERATOR_{op_id}', 'description': ''}
                for op_id in op_stats}

    print('\n  LLM response:\n' + response)

    cluster_names = {}
    for block in re.split(r'\n---\n?', response):
        m_id   = re.search(r'CLUSTER:\s*(\d+)', block)
        m_name = re.search(r'NAME:\s*(.+)',      block)
        m_desc = re.search(r'DESCRIPTION:\s*(.+)', block)
        if m_id and m_name:
            cid = int(m_id.group(1))
            cluster_names[cid] = {
                'name':        m_name.group(1).strip().upper(),
                'description': m_desc.group(1).strip() if m_desc else '',
            }

    for op_id in op_stats:
        if op_id not in cluster_names:
            cluster_names[op_id] = {'name': f'OPERATOR_{op_id}', 'description': ''}

    return cluster_names



def write_sample_trajectories(trace_spans, trace_meta, cluster_names, out_path, n=12):
    import random
    op_name     = {cid: v['name'] for cid, v in cluster_names.items()}
    op_name[-1] = 'STRUCTURAL'

    good = [tid for tid, spans in trace_spans.items()
            if len({s['cluster'] for s in spans if s['cluster'] >= 0}) >= 4
            and 15 <= len(spans) <= 80]

    random.seed(42)
    random.shuffle(good)
    seen, selected = set(), []
    for tid in good:
        key = (trace_meta[tid].get('model', ''), bool(trace_meta[tid].get('correct')))
        if key not in seen and len(selected) < n:
            seen.add(key)
            selected.append(tid)
    if len(selected) < n:
        selected += [t for t in good if t not in selected][:n - len(selected)]

    lines = []
    for tid in selected[:n]:
        spans = sorted(trace_spans[tid], key=lambda s: s['span_idx'])
        m     = trace_meta[tid]
        lines.append('=' * 80)
        lines.append(f'TRACE:   {tid}')
        lines.append(f'MODEL:   {m.get("model","?")}')
        lines.append(f'DATASET: {m.get("dataset","?")}')
        lines.append(f'CORRECT: {m.get("correct","?")}')
        lines.append('=' * 80)
        for sp in spans:
            label  = op_name.get(sp['cluster'], f'OP{sp["cluster"]}')
            anchor = sp['anchor'].replace('\n', ' ').strip()[:120]
            lines.append(f'[{label:<14}] {anchor}')
            for sent in sp['body'][:3]:
                lines.append(f'               {sent.replace(chr(10)," ").strip()[:110]}')
        lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Wrote {out_path}')



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',            required=True)
    ap.add_argument('--output_dir',       required=True)
    ap.add_argument('--e5_model',         default='intfloat/e5-small-v2')
    ap.add_argument('--min_trace_freq',   type=int, default=100)
    ap.add_argument('--min_domain_count', type=int, default=3)
    ap.add_argument('--top_vocab_size',   type=int, default=2000)
    ap.add_argument('--top_k_starts',     type=int, default=50_000)
    ap.add_argument('--n_operators',      type=int, default=10)
    ap.add_argument('--start_tokens',     type=int, default=3)
    ap.add_argument('--max_body',         type=int, default=50)
    ap.add_argument('--seed',             type=int, default=42)
    ap.add_argument('--skip_llm',         action='store_true')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    # 1. Load
    print('=' * 72)
    print('Step 1: Loading traces')
    traces = []
    opener = gzip.open if args.input.endswith('.gz') else open
    with opener(args.input, 'rt') as f:
        for line in f:
            d = json.loads(line)
            if KEEP_MODELS and d.get('model') not in KEEP_MODELS:
                continue
            traces.append(d)
    print(f'  {len(traces):,} traces')

    all_domains = sorted({d.get('dataset', 'unk') for d in traces})
    N_DOMAINS   = len(all_domains)
    print(f'  Domains ({N_DOMAINS}): {all_domains}')

    trace_meta = {}
    for d in traces:
        tid = (d.get('trace_id') or
               f'{d["model"]}__{d["dataset"]}__{d["problem_id"]}__{d.get("sample",0)}')
        d['_tid'] = tid
        trace_meta[tid] = {
            'model':      d.get('model', ''),
            'dataset':    d.get('dataset', ''),
            'problem_id': d.get('problem_id', ''),
            'sample':     d.get('sample', 0),
            'correct':    d.get('correct', None),
        }

    # 2. Sentence splitting
    print('\n' + '=' * 72)
    print('Step 2: Sentence splitting')
    trace_sentences = {}
    total_sents = 0
    for d in traces:
        tid   = d['_tid']
        sents = split_sentences(d.get('reasoning', ''))
        if sents:
            trace_sentences[tid] = sents
            total_sents += len(sents)
    print(f'  {total_sents:,} sentences across {len(trace_sentences):,} traces')

    # 3. Corpus vocabulary
    print('\n' + '=' * 72)
    print(f'Step 3: Building top-{args.top_vocab_size} corpus vocabulary')
    top_vocab = build_top_vocab(trace_sentences, top_n=args.top_vocab_size)

    # 4. Type-A starts (domain-filtered + vocab-filtered)
    print('\n' + '=' * 72)
    print('Step 4: Finding type-A sentence starts')
    type_a_starts = find_type_a_starts(
        trace_sentences, min_trace_freq=args.min_trace_freq,
        top_vocab=top_vocab, trace_meta=trace_meta,
        min_domain_count=args.min_domain_count,
        top_k=args.top_k_starts,
        n_start_tokens=args.start_tokens,
    )

    # 5. Segment
    print('\n' + '=' * 72)
    print('Step 5: Segmenting traces')
    trace_spans = segment_traces(trace_sentences, type_a_starts,
                                max_body=args.max_body,
                                n_start_tokens=args.start_tokens)
    total_spans = sum(len(v) for v in trace_spans.values())
    n_type_a    = sum(1 for spans in trace_spans.values()
                      for sp in spans if sp['anchor_reason'] == 'type_a')
    print(f'  {total_spans:,} spans')
    print(f'  Type-A anchored: {n_type_a:,} ({100*n_type_a/max(total_spans,1):.1f}%)')

    # 6. Load e5-small
    print('\n' + '=' * 72)
    print('Step 6: Loading e5-small')
    e5 = load_e5(args.e5_model)

    # 7. Cluster pivots → assign spans
    print('\n' + '=' * 72)
    print('Step 7: KMeans on type-A pivots (e5-small embeddings)')
    start_to_cluster = cluster_type_a_starts(
        type_a_starts, e5, n_operators=args.n_operators, seed=args.seed,
    )
    assign_span_clusters(trace_spans, start_to_cluster)

    op_stats = compute_op_stats(trace_spans, all_domains)
    print('\n  Cluster summary:')
    for op_id in sorted(op_stats):
        s = op_stats[op_id]
        print(f'    Cluster {op_id:2d}  n={s["n_sentences"]:7,}  '
              f'entropy={s["entropy"]:.3f}  tops: {s["top_starts"][:5]}')

    # 8. LLM naming
    print('\n' + '=' * 72)
    print('Step 8: LLM cluster naming')
    if args.skip_llm:
        cluster_names = {op_id: {'name': f'OPERATOR_{op_id}', 'description': ''}
                         for op_id in op_stats}
    else:
        cluster_names = name_clusters_llm(op_stats, trace_spans)

    print('\n  Final taxonomy:')
    for cid in sorted(cluster_names):
        cn = cluster_names[cid]
        ns = op_stats.get(cid, {}).get('n_sentences', 0)
        print(f'    [{cn["name"]:<16}] n={ns:,}')

    # 9. Write outputs
    print('\n' + '=' * 72)
    print('Step 9: Writing outputs')

    with open(out / 'cluster_names.json', 'w') as f:
        json.dump({str(k): v for k, v in cluster_names.items()}, f, indent=2)

    with open(out / 'operator_vocab.txt', 'w') as f:
        for op_id in sorted(op_stats):
            name = cluster_names.get(op_id, {}).get('name', f'OP_{op_id}')
            f.write(f'# {name}\n')
            for s in op_stats[op_id]['top_starts'][:15]:
                f.write(s + '\n')

    with open(out / 'spans_clustered.jsonl', 'w') as fout:
        for d in tqdm(traces, desc='Writing spans', unit='trace'):
            tid  = d['_tid']
            meta = trace_meta[tid]
            for sp in trace_spans.get(tid, []):
                fout.write(json.dumps({
                    'trace_id':      tid,
                    'model':         meta['model'],
                    'dataset':       meta['dataset'],
                    'problem_id':    meta['problem_id'],
                    'sample':        meta['sample'],
                    'anchor':        sp['anchor'],
                    'body':          sp['body'],
                    'body_len':      sp['body_len'],
                    'span_idx':      sp['span_idx'],
                    'anchor_reason': sp['anchor_reason'],
                    'ngram':         sp.get('ngram', ''),
                    'cluster':       sp['cluster'],
                    'correct':       meta['correct'],
                }) + '\n')
    print(f'  spans_clustered.jsonl  ({total_spans:,} spans)')

    by_op    = defaultdict(int)
    n_struct = 0
    for spans in trace_spans.values():
        for sp in spans:
            if sp['cluster'] >= 0:
                by_op[sp['cluster']] += 1
            else:
                n_struct += 1

    with open(out / 'clusters.json', 'w') as f:
        json.dump({
            'n_operators':   args.n_operators,
            'total_spans':   total_spans,
            'structural':    n_struct,
            'by_operator':   {str(k): v for k, v in sorted(by_op.items())},
            'cluster_names': {str(k): v for k, v in cluster_names.items()},
            'op_stats':      {str(k): v for k, v in op_stats.items()},
        }, f, indent=2)

    print('\n' + '=' * 60)
    print(f'  STRUCTURAL    {n_struct:>10,} spans')
    for op_id in sorted(cluster_names):
        name = cluster_names[op_id]['name']
        n    = by_op.get(op_id, 0)
        print(f'  {name:<20}{n:>10,} spans')
    print('=' * 60)

    write_sample_trajectories(
        trace_spans, trace_meta, cluster_names,
        out / 'labeled_trajectories_sample.txt',
    )

    print(f'\nDone. Outputs -> {out}/')


if __name__ == '__main__':
    main()
