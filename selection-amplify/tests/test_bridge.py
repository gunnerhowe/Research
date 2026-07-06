import numpy as np

from selamp import data
from selamp.bridge import RewardConfig, generate_labeled, guided_sample
from selamp.diffusion import Diffusion
from selamp.selection import SelectionEstimator
from selamp.validate import GateKDE, IndependentValidator


def _setup(beta=4.0, seed=0):
    c = data.make_corpora("two_moons", beta=beta, seed=seed, n_pop=8000)
    est = SelectionEstimator(n_members=3, epochs=200).fit(
        c.X_obs, c.X_ref, c.obs_frac, seed=seed)
    dm = Diffusion(T=120).fit(c.X_obs, c.y_obs, epochs=1500, seed=seed)
    gate = GateKDE(c.X_ref, bandwidth=0.3)
    return c, est, dm, gate


def test_guidance_lowers_selection_vs_unguided():
    c, est, dm, gate = _setup()
    cfg = RewardConfig(gamma=6.0)
    Xg, _ = guided_sample(dm, est, gate, 0, 800, cfg, seed=1)
    cfg0 = RewardConfig(gamma=0.0)
    X0, _ = guided_sample(dm, est, gate, 0, 800, cfg0, seed=1)
    # guided samples sit at LOWER true selection (deeper in the complement)
    s_g = data.selection_prob(Xg, c.beta, c.testbed).mean()
    s_0 = data.selection_prob(X0, c.beta, c.testbed).mean()
    assert s_g < s_0 - 0.02


def test_guidance_stays_on_manifold():
    c, est, dm, gate = _setup()
    val = IndependentValidator(np.concatenate(
        [data.two_moons(8000, seed=99)[0]]))
    cfg = RewardConfig(gamma=6.0)
    X, _, _ = generate_labeled(dm, est, gate, 800, cfg, seed=1)
    rej = val.score(X)["reject_rate"]
    assert rej < 0.25                       # off-manifold guard holds (K3)


def test_decoy_differs_from_method():
    c, est, dm, gate = _setup()
    cfg = RewardConfig(gamma=6.0)
    Xm, _ = guided_sample(dm, est, gate, 0, 800, cfg, decoy=None, seed=1)
    Xd, _ = guided_sample(dm, est, gate, 0, 800, cfg, decoy="rotate", seed=1)
    # method drives to lower true selection than the misdirected decoy
    s_m = data.selection_prob(Xm, c.beta, c.testbed).mean()
    s_d = data.selection_prob(Xd, c.beta, c.testbed).mean()
    assert s_m < s_d


def test_guidance_norm_cap():
    c, est, dm, gate = _setup()
    cfg = RewardConfig(gamma=50.0, cap_ratio=1.0)   # huge gamma -> must cap
    _, diag = guided_sample(dm, est, gate, 0, 400, cfg, seed=1)
    assert diag.mean_guide_norm <= diag.mean_base_norm + 1e-5
