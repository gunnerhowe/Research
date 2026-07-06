"""End-to-end sanity of the sequential trainer and metrics."""
import numpy as np

from doobsyn.data import yin_yang, split_mnist_domain
from doobsyn.sim import run_sequence


def test_yin_yang_shapes_and_labels():
    tasks = yin_yang(n_tasks=3, n_train=300, n_test=200, seed=0)
    assert len(tasks) == 3
    for t in tasks:
        assert t["Xtr"].shape[1] == 4
        assert set(np.unique(t["ytr"].numpy()).tolist()).issubset({0, 1, 2})


def test_run_sequence_metrics_in_range():
    tasks = yin_yang(n_tasks=3, n_train=400, n_test=200, seed=0)
    r = run_sequence("yin_yang", tasks, method="doob", sigma=0.02, seed=0,
                     epochs=1, batch_size=64, device="cpu")
    assert 0.0 <= r["retention"] <= 1.0
    assert 0.0 <= r["avg_acc"] <= 1.0
    assert np.array(r["A"]).shape == (3, 3)
    assert r["n_params"] > 0


def test_doob_equals_ou_at_zero_noise_end_to_end():
    """With identical seed and sigma=0, doob and ou must produce the SAME run
    (the Doob term is off), a strong end-to-end determinism check."""
    tasks = yin_yang(n_tasks=3, n_train=400, n_test=200, seed=0)
    kw = dict(sigma=0.0, seed=0, epochs=1, batch_size=64, device="cpu")
    rd = run_sequence("yin_yang", tasks, method="doob", **kw)
    ru = run_sequence("yin_yang", tasks, method="ou", **kw)
    assert abs(rd["retention"] - ru["retention"]) < 1e-9


def test_split_mnist_task_structure():
    tasks = split_mnist_domain(n_per_class_train=50, seed=0)
    assert len(tasks) == 5
    for t in tasks:
        assert t["Xtr"].shape[1] == 784
        assert set(np.unique(t["ytr"].numpy()).tolist()).issubset({0, 1})
