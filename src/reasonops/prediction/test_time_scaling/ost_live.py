"""Segment generated traces and score operator sequences with a saved OST."""

from reasonops.operators.discover_operators import sentence_start, split_sentences
from reasonops.operators.pivot_dictionary import load_pivots

_PIV = None


def segment(text):
    global _PIV
    if _PIV is None:
        _PIV = load_pivots()
    seq = []
    for s in split_sentences(text or ""):
        st = sentence_start(s, n=3)
        if st:
            c = _PIV.get(" ".join(st))
            if c is not None:
                seq.append(c)
    return seq


class OSTScorer:
    def __init__(self, path, device=None):
        import torch

        from reasonops.prediction.seq_pred import OST

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device)
        c = ck.get("config", {})
        self.model = OST(
            d_model=c.get("d_model", 128),
            n_heads=c.get("n_heads", 4),
            n_layers=c.get("n_layers", 4),
        )
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        self.device = device

    def score(self, seq):
        return self.score_batch([seq])[0]

    def score_batch(self, seqs, chunk=512):
        import torch

        probs = [0.5] * len(seqs)
        idx = [i for i, s in enumerate(seqs) if s]
        for c in range(0, len(idx), chunk):
            js = idx[c : c + chunk]
            batch = [seqs[j][:512] for j in js]
            L = max(len(seq) for seq in batch)
            ids = torch.full((len(batch), L), 7, dtype=torch.long)
            for r, s in enumerate(batch):
                ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            with torch.no_grad():
                logit, _ = self.model(ids.to(self.device))
                p = torch.sigmoid(logit).cpu().tolist()
            for r, j in enumerate(js):
                probs[j] = float(p[r])
        return probs
