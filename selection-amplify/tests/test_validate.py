import numpy as np

from selamp import data
from selamp.validate import GateKDE, IndependentValidator


def test_gate_kde_higher_on_data():
    X, _ = data.two_moons(4000, seed=0)
    gate = GateKDE(X, bandwidth=0.3)
    on = gate.log_p_np(X[:500]).mean()
    off = gate.log_p_np(X[:500] + np.array([10.0, 10.0])).mean()
    assert on > off + 5


def test_independent_validator_flags_offmanifold():
    X, _ = data.two_moons(6000, seed=0)
    val = IndependentValidator(X)
    good = val.score(data.two_moons(1000, seed=1)[0])["reject_rate"]
    bad = val.score(X[:1000] + np.array([8.0, 8.0]))["reject_rate"]
    assert good < 0.2 and bad > 0.9


def test_quantile_threshold_monotone():
    X, _ = data.two_moons(4000, seed=0)
    gate = GateKDE(X, bandwidth=0.3)
    t10 = gate.quantile_threshold(X, 0.10)
    t50 = gate.quantile_threshold(X, 0.50)
    assert t10 < t50
