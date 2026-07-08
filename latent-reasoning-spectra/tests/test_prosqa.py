import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.paths import DATA  # noqa: E402
from lrspec.prosqa import load_problems, prune_to_linear  # noqa: E402


def test_paths_and_labels():
    ps = load_problems(DATA / "prosqa_test.json")  # asserts path/DAG consistency inside
    p = ps[0]
    assert p.path[0] == p.root and p.path[-1] == p.target
    assert p.branch_label(1) in (0, 1)
    assert p.branch_label(p.n_hops + 1) is None  # padding step


def test_prune_to_linear():
    ps = load_problems(DATA / "prosqa_test.json")
    n_ok = 0
    for p in ps[:50]:
        q = prune_to_linear(p)
        if q is None:
            continue
        n_ok += 1
        assert q.is_linear_chain
        assert q.path == p.path  # same ground-truth path
        assert q.question.endswith(p.question.split(". ")[-1])  # same final question
        # all remaining statements are a subset of the original ones
        orig = set(p.question[: p.question.rfind(".") + 1].split(". "))
        for s in q.question.split(". ")[:-1]:
            assert s.rstrip(".") in {o.rstrip(".") for o in orig}
        # neg target still mentioned
        assert f" {p.idx_to_symbol[p.neg_target]}" in q.question
    assert n_ok >= 40  # most instances should survive the neg-target guard
