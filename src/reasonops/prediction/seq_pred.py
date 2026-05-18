#!/usr/bin/env python3
"""
Operator Sequence Transformer (OST) for correctness prediction.

A lightweight Transformer encoder over discrete operator label sequences.
Per position the input is a sum of unigram + bigram (transition) embeddings
plus a continuous sinusoidal position encoding scaled by normalized position.
Trained with a pure pairwise contrastive loss within each problem, with
5% token dropout, attention pooling, and WP-AUC early stopping.

Usage:
    python -m reasonops.prediction.seq_pred \
        --mode       cv \
        --corpus     data/final_dataset.jsonl.gz \
        --output_dir data/ost_cv
"""
import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from reasonops.utils import EXCLUDE_DATASETS, N_FOLDS, SEED, FRACS

N_OPS = 7
PAD_ID = N_OPS          # padding token id
VOCAB  = N_OPS + 1      # 0–6 operators + padding
TOKEN_DROPOUT_P = 0.05

class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, h, mask=None):
        scores = self.w(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * w).sum(dim=1)


class OST(nn.Module):
    """Operator Sequence Transformer.

    Architecture:
      - Unigram operator embedding + bigram (transition) embedding per position.
      - Continuous sinusoidal position encoding of pos/seq_len in [0, 1].
      - Pre-LayerNorm Transformer encoder.
      - Attention pooling → linear head → scalar logit.
    """
    def __init__(self, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        K = N_OPS
        self.K            = K
        self.BG_START     = K * K          # bigram id for "no previous token"
        self.tok_embed    = nn.Embedding(K + 1,     d_model, padding_idx=K)
        self.bigram_embed = nn.Embedding(K * K + 1, d_model)
        self.drop_in      = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer  = nn.TransformerEncoder(enc, n_layers, norm=nn.LayerNorm(d_model))
        self.pool         = AttentionPool(d_model)
        self.head         = nn.Linear(d_model, 1)

    def _pos_enc(self, x):
        B, L = x.shape
        d    = self.tok_embed.embedding_dim
        lens = (~(x == self.K)).sum(dim=1, keepdim=True).float().clamp(min=1)
        pos  = torch.arange(L, device=x.device).float().unsqueeze(0).expand(B, -1)
        norm = (pos / (lens - 1).clamp(min=1)).clamp(0, 1)
        div  = torch.exp(-torch.arange(0, d, 2, device=x.device).float() * math.log(10000.0) / d)
        sc   = norm.unsqueeze(-1) * math.pi * div
        pe   = torch.zeros(B, L, d, device=x.device)
        pe[..., 0::2] = torch.sin(sc)
        pe[..., 1::2] = torch.cos(sc)
        return pe

    def encode(self, x):
        K   = self.K
        pad = (x == K)
        x_s = x.clamp(0, K - 1)
        bg  = torch.full(x.shape, self.BG_START, dtype=torch.long, device=x.device)
        if x.shape[1] > 1:
            bg[:, 1:] = x_s[:, :-1] * K + x_s[:, 1:]
        bg.masked_fill_(pad, self.BG_START)
        bg_emb = self.bigram_embed(bg).masked_fill(pad.unsqueeze(-1), 0.0)
        h = self.drop_in(self.tok_embed(x) + bg_emb + self._pos_enc(x))
        h = self.transformer(h, src_key_padding_mask=pad)
        return self.pool(h, pad)

    def forward(self, x, lengths=None):
        z = self.encode(x)
        return self.head(z).squeeze(-1), z



def load_corpus(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dataset", "") in EXCLUDE_DATASETS:
                continue
            if r.get("correct") is None:
                continue
            seq = [int(x) for x in r.get("operator_sequence", [])]
            if not seq:
                continue
            rows.append({
                "trace_id":   r["trace_id"],
                "dataset":    r.get("dataset", ""),
                "model":      r.get("model", ""),
                "problem_id": r.get("problem_id", ""),
                "correct":    int(bool(r["correct"])),
                "seq":        seq,
            })
    return rows


def collate(batch, max_len, device):
    seqs = [r["seq"][:max_len] for r in batch]
    L    = max(len(s) for s in seqs)
    ids  = torch.full((len(seqs), L), PAD_ID, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    labels = torch.tensor([r["correct"] for r in batch], dtype=torch.float32, device=device)
    return ids, labels


def partial_collate(batch, frac, max_len, device):
    seqs = [r["seq"][:max(1, int(len(r["seq"]) * frac))][:max_len] for r in batch]
    L    = max(len(s) for s in seqs)
    ids  = torch.full((len(seqs), L), PAD_ID, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    labels = torch.tensor([r["correct"] for r in batch], dtype=torch.float32, device=device)
    return ids, labels


def problem_kfolds(rows, k=5, seed=42):
    pids = sorted(set(r["problem_id"] for r in rows))
    rng  = random.Random(seed)
    rng.shuffle(pids)
    pid_folds = [set(pids[i::k]) for i in range(k)]
    folds = []
    for fi in range(k):
        te = [r for r in rows if r["problem_id"] in     pid_folds[fi]]
        tr = [r for r in rows if r["problem_id"] not in pid_folds[fi]]
        folds.append((tr, te))
    return folds


def build_contrast_pairs(rows, max_per_problem=16):
    by_pid = defaultdict(lambda: {"pos": [], "neg": []})
    for r in rows:
        by_pid[r["problem_id"]]["pos" if r["correct"] else "neg"].append(r)
    pairs = []
    rng   = random.Random(42)
    for pid, d in by_pid.items():
        if not d["pos"] or not d["neg"]:
            continue
        pos_list = d["pos"] * (max_per_problem // max(len(d["pos"]), 1) + 1)
        neg_list = d["neg"] * (max_per_problem // max(len(d["neg"]), 1) + 1)
        rng.shuffle(pos_list)
        rng.shuffle(neg_list)
        for p, n in zip(pos_list[:max_per_problem], neg_list[:max_per_problem]):
            pairs.append((p, n))
    return pairs



def eval_model(model, rows, device, args):
    model.eval()
    all_scores, all_y, all_pid = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            ids, _ = collate(batch, args.max_len, device)
            scores, _ = model(ids)
            all_scores.extend(torch.sigmoid(scores).cpu().tolist())
            all_y.extend(r["correct"] for r in batch)
            all_pid.extend(r["problem_id"] for r in batch)
    try:
        pooled = roc_auc_score(all_y, all_scores)
    except Exception:
        pooled = 0.5
    groups = defaultdict(list)
    for pid, yt, yp in zip(all_pid, all_y, all_scores):
        groups[pid].append((yt, yp))
    aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
            for v in groups.values() if len(set(p[0] for p in v)) == 2]
    wp = float(np.mean(aucs)) if aucs else float("nan")
    return pooled, wp, len(aucs)


def collect_oof(model, rows, device, args):
    """Run the model at each FRACS depth on `rows` and return OOF prediction dicts."""
    model.eval()
    results = []
    with torch.no_grad():
        for frac in FRACS:
            scores_frac = []
            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                ids, _ = partial_collate(batch, frac, args.max_len, device)
                s, _ = model(ids)
                scores_frac.extend(torch.sigmoid(s).cpu().tolist())
            for j, r in enumerate(rows):
                if frac == FRACS[0]:
                    results.append({k: r[k] for k in
                                    ("trace_id", "dataset", "model", "problem_id", "correct")})
                results[j][f"d{int(frac*100)}"] = scores_frac[j]
    return results


def _token_dropout(ids, p, device):
    mask = (ids != PAD_ID) & (torch.rand(ids.shape, device=device) < p)
    return ids.masked_fill(mask, PAD_ID)


def train_fold(model, train_rows, val_rows, device, args):
    """One-fold training with pairwise contrastive loss + token dropout.

    Selects the best checkpoint by WP-AUC on the validation split when
    n_problems >= 10, else by pooled AUC (early-stopping on the more reliable
    of the two for the validation size at hand).
    """
    pairs  = build_contrast_pairs(train_rows, args.contrast_k)
    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best_m, best_state, no_improve = -1.0, None, 0
    checkpoint_path = Path(args.output_dir) / "ost_tmp.pt"

    for ep in range(args.epochs):
        model.train()
        rng = random.Random(args.seed + ep)
        rng.shuffle(pairs)
        total_c = nb = 0.0
        for i in range(0, len(pairs), args.batch_size // 2):
            batch_pairs = pairs[i:i + args.batch_size // 2]
            if not batch_pairs:
                continue
            pos_ids, _ = collate([p for p, _ in batch_pairs], args.max_len, device)
            neg_ids, _ = collate([n for _, n in batch_pairs], args.max_len, device)
            pos_ids = _token_dropout(pos_ids, TOKEN_DROPOUT_P, device)
            neg_ids = _token_dropout(neg_ids, TOKEN_DROPOUT_P, device)
            pos_s, _ = model(pos_ids)
            neg_s, _ = model(neg_ids)
            loss = F.softplus(neg_s - pos_s).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_c += loss.item(); nb += 1
        sched.step()
        pooled, wp, wp_n = eval_model(model, val_rows, device, args)
        wp_safe = wp if not math.isnan(wp) else 0.0
        metric  = wp_safe if wp_n >= 10 else pooled
        print(f"  ep{ep:2d}: loss_c={total_c/max(nb,1):.4f}"
              f" | pooled={pooled:.4f} | wp_auc={wp:.4f}(n={wp_n})"
              f" lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if metric > best_m:
            best_m     = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save({"state_dict": best_state, "epoch": ep}, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop ep{ep}", flush=True)
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_m



def make_model(args):
    return OST(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)


def print_summary(all_oof):
    all_y   = [r["correct"]    for r in all_oof]
    all_p   = [r["d100"]       for r in all_oof]
    all_pid = [r["problem_id"] for r in all_oof]
    groups  = defaultdict(list)
    for pid, yt, yp in zip(all_pid, all_y, all_p):
        groups[pid].append((yt, yp))
    aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
            for v in groups.values() if len(set(p[0] for p in v)) == 2]
    print(f"Global WP-AUC (@100%): {float(np.mean(aucs)):.4f}  (n_problems={len(aucs)})", flush=True)
    by_ds = defaultdict(lambda: {"y": [], "p": [], "pid": []})
    for r in all_oof:
        by_ds[r["dataset"]]["y"].append(r["correct"])
        by_ds[r["dataset"]]["p"].append(r["d100"])
        by_ds[r["dataset"]]["pid"].append(r["problem_id"])
    for ds in sorted(by_ds):
        d = by_ds[ds]
        if len(set(d["y"])) < 2: continue
        grps = defaultdict(list)
        for pid, yt, yp in zip(d["pid"], d["y"], d["p"]):
            grps[pid].append((yt, yp))
        ds_aucs = [roc_auc_score([p[0] for p in v], [p[1] for p in v])
                   for v in grps.values() if len(set(p[0] for p in v)) == 2]
        print(f"  {ds:<20} {float(np.mean(ds_aucs)):.4f} (n={len(ds_aucs)})", flush=True)


def _cv_loop(args, rows):
    device    = torch.device(args.device)
    folds     = problem_kfolds(rows, k=args.k_folds, seed=args.seed)
    all_oof   = []
    for fold_idx, (train_all, test_all) in enumerate(folds):
        print(f"\n{'='*60}\nFOLD {fold_idx+1}/{args.k_folds}"
              f"  train={len(train_all):,}  test={len(test_all):,}\n{'='*60}")
        train_pids = sorted(set(r["problem_id"] for r in train_all))
        rng = random.Random(args.seed + fold_idx)
        rng.shuffle(train_pids)
        val_pids = set(train_pids[:max(1, len(train_pids) // 10)])
        train = [r for r in train_all if r["problem_id"] not in val_pids]
        val   = [r for r in train_all if r["problem_id"] in val_pids]
        print(f"  train={len(train):,}  val={len(val):,}  test={len(test_all):,}", flush=True)
        model = make_model(args).to(device)
        print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)
        train_fold(model, train, val, device, args)
        oof = collect_oof(model, test_all, device, args)
        all_oof.extend(oof)
        _, wp, wp_n = eval_model(model, test_all, device, args)
        print(f"  Fold test wp_auc={wp:.4f}(n={wp_n})", flush=True)
    return all_oof


def _within_loop(args, rows):
    device  = torch.device(args.device)
    all_oof = []
    for ds in sorted(set(r["dataset"] for r in rows)):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if len(ds_rows) < 20 or len(set(r["correct"] for r in ds_rows)) < 2:
            continue
        print(f"\n{'='*60}\nDataset: {ds}  ({len(ds_rows):,} traces)\n{'='*60}", flush=True)
        folds = problem_kfolds(ds_rows, k=args.k_folds, seed=args.seed)
        for fold_idx, (train_all, test_all) in enumerate(folds):
            if not train_all or not test_all:
                continue
            train_pids = sorted(set(r["problem_id"] for r in train_all))
            rng = random.Random(args.seed + fold_idx)
            rng.shuffle(train_pids)
            val_pids = set(train_pids[:max(1, len(train_pids) // 10)])
            train = [r for r in train_all if r["problem_id"] not in val_pids]
            val   = [r for r in train_all if r["problem_id"] in val_pids]
            if len(set(r["correct"] for r in train)) < 2:
                print(f"  Fold {fold_idx+1}: skipping (single class)", flush=True)
                continue
            print(f"\n  Fold {fold_idx+1}/{args.k_folds}: "
                  f"train={len(train):,}  val={len(val):,}  test={len(test_all):,}", flush=True)
            model = make_model(args).to(device)
            train_fold(model, train, val, device, args)
            all_oof.extend(collect_oof(model, test_all, device, args))
        ds_oof = [r for r in all_oof if r["dataset"] == ds]
        if ds_oof:
            grps = defaultdict(list)
            for r in ds_oof:
                grps[r["problem_id"]].append((r["correct"], r["d100"]))
            ds_aucs = [roc_auc_score([x[0] for x in v], [x[1] for x in v])
                       for v in grps.values() if len(set(x[0] for x in v)) == 2]
            if ds_aucs:
                print(f"  {ds} WP-AUC = {float(np.mean(ds_aucs)):.4f} (n={len(ds_aucs)})", flush=True)
    return all_oof


def _save_oof(all_oof, args, suffix=""):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"preds_ost{suffix}.jsonl.gz"
    with gzip.open(out / fname, "wt") as f:
        for r in all_oof:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(all_oof):,} OOF predictions → {out / fname}", flush=True)
    print_summary(all_oof)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="cv", choices=["cv", "within"],
                    help="cv: cross-dataset 5-fold; within: per-dataset 5-fold")
    ap.add_argument("--corpus",     required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs",     type=int,   default=60)
    ap.add_argument("--patience",   type=int,   default=12)
    ap.add_argument("--batch_size", type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--max_len",    type=int,   default=512)
    ap.add_argument("--contrast_k", type=int,   default=16)
    ap.add_argument("--k_folds",    type=int,   default=5)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--d_model",    type=int,   default=128)
    ap.add_argument("--n_heads",    type=int,   default=4)
    ap.add_argument("--n_layers",   type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading corpus: {args.corpus}", flush=True)
    rows = load_corpus(args.corpus)
    print(f"  {len(rows):,} traces | mode={args.mode}", flush=True)

    if args.mode == "within":
        all_oof = _within_loop(args, rows)
        _save_oof(all_oof, args, suffix="_within")
    else:
        all_oof = _cv_loop(args, rows)
        _save_oof(all_oof, args)


if __name__ == "__main__":
    main()
