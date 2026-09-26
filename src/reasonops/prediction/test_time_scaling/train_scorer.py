"""Train an OST while excluding the evaluation problems."""

import random
from types import SimpleNamespace

import torch

from reasonops.paths import DATA_DIR
from reasonops.prediction.seq_pred import load_corpus, make_model, train_fold

from .config import HERE, held_out_ids


def pool_problem_ids():
    return held_out_ids()


DEPTHS = [0.25, 0.5, 0.75, 1.0]


def prefix_views(rows):
    out = []
    for r in rows:
        for d in DEPTHS:
            k = max(1, int(len(r["seq"]) * d))
            out.append({**r, "seq": r["seq"][:k], "problem_id": f"{r['problem_id']}@{d}"})
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-aug", action="store_true")
    parser.add_argument("--in-domain", action="store_true")
    cli = parser.parse_args()
    prefix_aug, in_domain = (cli.prefix_aug, cli.in_domain)
    HERE.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        output_dir=str(HERE),
        epochs=60,
        patience=12,
        batch_size=128,
        lr=0.0003,
        max_len=512,
        contrast_k=16,
        seed=42,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    held_out = pool_problem_ids()
    rows = [
        r
        for r in load_corpus(DATA_DIR / "final_dataset.jsonl.gz")
        if str(r["problem_id"]) not in held_out
    ]
    if in_domain:
        rows = [r for r in rows if r["dataset"] in ("aime", "livecodebench")]
    print(
        f"device={args.device}  {len(rows):,} traces after holding out {len(held_out)} pool problems (in_domain={in_domain})",
        flush=True,
    )
    pids = sorted(set((r["problem_id"] for r in rows)))
    rng = random.Random(args.seed)
    rng.shuffle(pids)
    val_pids = set(pids[: max(1, len(pids) // 10)])
    train = [r for r in rows if r["problem_id"] not in val_pids]
    val = [r for r in rows if r["problem_id"] in val_pids]
    if prefix_aug:
        train, val = (prefix_views(train), prefix_views(val))
    print(f"train={len(train):,}  val={len(val):,}  prefix_aug={prefix_aug}", flush=True)
    model = make_model(args).to(args.device)
    best = train_fold(model, train, val, args.device, args)
    print(f"best val metric = {best:.4f}", flush=True)
    name = (
        "ost_scorer"
        + ("_prefix" if prefix_aug else "_holdout")
        + ("_indomain" if in_domain else "")
        + ".pt"
    )
    out = HERE / name
    torch.save(
        {"state_dict": model.state_dict(), "config": {"d_model": 128, "n_heads": 4, "n_layers": 4}},
        out,
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
