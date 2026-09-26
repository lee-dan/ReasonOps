import json

import numpy as np
import pytest

from reasonops.eval import system_prompting
from reasonops.eval.sensitivity import sweep
from reasonops.operators.pivot_dictionary import build_pivots
from reasonops.prediction.op_seq_baseline import op_features
from reasonops.prediction.test_time_scaling import (
    best_of_n,
    config,
    ost_live,
    score_pool,
    selection,
)


@pytest.mark.parametrize("seq", [[], [0], [6, 6], [0, 1, 2], list(range(7)) * 5])
def test_variable_k_features_match_original(seq):
    np.testing.assert_array_equal(sweep.op_features(seq, 7), op_features(seq))
    assert sweep.op_features(seq, 7).shape == (117,)


def test_sweep_grid_and_filters():
    cells = sweep.start_cells(sweep.START_AXES, [42, 1, 2])
    assert len([c for c in cells if c["axis"] == "joint"]) == 24
    assert cells[0] == dict(axis="default", seed=42, **sweep.DEFAULT)
    traces = {
        "a": {"dataset": "math", "starts": [("let", "me", "think")] * 10},
        "b": {"dataset": "gpqa", "starts": [("let", "me", "think")]},
    }
    vocab = {"let", "me", "think"}
    assert sweep.accepted_pivots(traces, 3, 2, 2, vocab) == {("let", "me", "think")}
    assert not sweep.accepted_pivots(traces, 3, 3, 2, vocab)
    assert not sweep.accepted_pivots(traces, 3, 2, 3, vocab)


def test_dictionary_and_live_segmentation(tmp_path, monkeypatch):
    rows = [
        {"ngram": "let me think", "cluster": 0},
        {"ngram": "so the answer", "cluster": 3},
        {"ngram": "", "cluster": -1},
    ]
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    pivots = build_pivots(path)
    monkeypatch.setattr(ost_live, "_PIV", pivots)
    monkeypatch.setattr(system_prompting, "load_pivots", lambda: pivots)
    text = "Let me think carefully. Unknown opening.\n- A list item\nSo the answer is 42."
    assert ost_live.segment(text) == [0, 3]
    assert system_prompting.featurize([text])[0] == [[0, 3]]
    rows.append({"ngram": "let me think", "cluster": 2})
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="Inconsistent"):
        build_pivots(path)


def test_pilot_preserves_full_holdout(tmp_path, monkeypatch):
    path = tmp_path / "problems.json"
    path.write_text(json.dumps({"aime": [{"problem_id": "a"}, {"problem_id": "b"}]}))
    monkeypatch.setattr(config, "PROBLEMS_JSON", path)
    monkeypatch.setenv("TTS_PILOT", "1")
    assert len(config.problems()["aime"]) == 1
    assert config.held_out_ids() == {"a", "b"}


def test_selection_denominators_and_ties():
    attempts = [
        dict(answer="1", correct=True, chars=1, op=0.8),
        dict(answer="2", correct=False, chars=2, op=0.2),
    ]
    assert selection.tally(attempts) == 0.5
    out = selection.policies(attempts, np.random.default_rng(0), voting=True)
    assert out["op"] == out["oracle"] == 1
    assert out["random"] == 0.5
    assert all(v <= out["oracle"] for v in out.values())
    for a in attempts:
        a["answer"] = None
    assert selection.tally(attempts) == 0


def test_random_best_of_n_stays_at_base_rate():
    cells = {
        ("aime", "model", "p"): [{"correct": int(i >= 50), "op": float(i)} for i in range(100)]
    }
    random = best_of_n.bon_per_cell(cells, None, np.random.default_rng(4), 12, random_key=True)
    perfect = best_of_n.bon_per_cell(cells, lambda a: a["op"], np.random.default_rng(4), 12)
    assert abs(next(iter(random.values())) - 0.5) < 0.04
    assert next(iter(perfect.values())) > 0.99


def test_problem_bootstrap_not_attempt_weighted():
    a = {("aime", "m1", "p1"): 1.0, ("aime", "m2", "p1"): 1.0, ("aime", "m1", "p2"): 0.0}
    b = {key: 0.0 for key in a}
    result = best_of_n.paired_over_problems(a, b)
    assert result["n"] == 2
    assert result["mean"] == 0.5


def test_out_of_fold_problems_are_disjoint(monkeypatch):
    from reasonops import utils

    seen = []

    class Classifier:
        def fit(self, x, y):
            self.train_problems = set(x[:, 0])

        def predict_proba(self, x):
            assert self.train_problems.isdisjoint(x[:, 0])
            seen.extend(x[:, 0])
            return np.tile([0.5, 0.5], (len(x), 1))

    monkeypatch.setattr(utils, "make_clf", Classifier)
    rows = [dict(problem_id=str(p), correct=y, problem_number=p) for p in range(4) for y in [0, 1]]
    result = score_pool.fit_oof(rows, lambda r: [r["problem_number"]])
    assert len(seen) == len(rows)
    np.testing.assert_array_equal(result, np.full(len(rows), 0.5))


def test_ost_checkpoint_inference(tmp_path):
    import torch

    from reasonops.prediction.seq_pred import OST

    torch.manual_seed(0)
    model = OST(d_model=16, n_heads=4, n_layers=1)
    path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "config": dict(d_model=16, n_heads=4, n_layers=1)}, path
    )
    scorer = ost_live.OSTScorer(path, device="cpu")
    scores = scorer.score_batch([[], [0], list(range(7))])
    assert scores[0] == 0.5
    assert all(0 <= v <= 1 for v in scores)


def test_human_agreement_keeps_abstentions():
    from reasonops.eval.human_validation import compare, normalize

    rows = [
        dict(pipeline_label="INITIATING", annotator_1="INITIATING", annotator_2=""),
        dict(pipeline_label="GROUNDING", annotator_1="NONE", annotator_2="GROUNDING"),
    ]
    assert normalize("bac") == "BACKTRACKING"
    a1 = compare(rows, "pipeline_label", "annotator_1")
    a2 = compare(rows, "pipeline_label", "annotator_2")
    assert a1["n"] == 2 and a1["agreement"] == 0.5
    assert a1["confusion"][2][7] == 1
    assert a2["n"] == 1 and a2["agreement"] == 1
    assert a2["kappa"] is None


def test_collect_trace_ids_match_assembly(tmp_path):
    from reasonops.data.collect import records

    path = tmp_path / "sample_2" / "qwen3-30b-thinking" / "aime" / "1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(problem_id="1", reasoning="Let me think.", correct=True)))
    (row,) = records(tmp_path)
    assert row["trace_id"] == "qwen3-30b-thinking__aime__1__s2"
    assert row["dataset"] == "aime" and row["sample"] == 2


def test_nested_corpus_config(tmp_path):
    from reasonops.data.filter import load_config

    path = tmp_path / "data.toml"
    path.write_text('[corpus]\ndropped_models = ["excluded"]\n')
    assert load_config(path)["dropped_models"] == ["excluded"]
