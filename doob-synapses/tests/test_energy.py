"""The operation-count energy model and the noise tax."""
from doobsyn.energy import (step_energy_pj, noise_tax_ratio, consolidation_cost)


def test_noise_tax_positive_only_with_noise():
    assert noise_tax_ratio("doob", 10000, 0.0) == 0.0
    assert noise_tax_ratio("doob", 10000, 0.1) > 0.0
    assert noise_tax_ratio("ou", 10000, 0.1) > 0.0


def test_bss2_cheaper_than_gpu():
    for method in ("ou", "doob", "mesu"):
        g = step_energy_pj(method, 50000, 0.1, substrate="gpu")
        b = step_energy_pj(method, 50000, 0.1, substrate="bss2")
        assert b < g                              # analog + free noise is cheaper


def test_doob_costs_more_than_ou_on_gpu():
    """The extra transcendental (tan) makes the Doob consolidation step cost more
    than a plain anchored step on a digital accelerator -- an honest overhead we
    report (the payoff is the noise being free on silicon)."""
    cd = consolidation_cost("doob", 10000, 0.1)
    cu = consolidation_cost("ou", 10000, 0.1)
    assert cd.energy_pj() > cu.energy_pj()
    assert cd.tanh > 0 and cu.tanh == 0


def test_bss2_drops_the_rng():
    """On silicon the diffusion noise is intrinsic: no RNG draw is charged."""
    g = step_energy_pj("doob", 20000, 0.2, substrate="gpu")
    g0 = step_energy_pj("doob", 20000, 0.0, substrate="gpu")
    b = step_energy_pj("doob", 20000, 0.2, substrate="bss2")
    b0 = step_energy_pj("doob", 20000, 0.0, substrate="bss2")
    assert (g - g0) > 0                            # gpu pays for noise
    assert abs(b - b0) < 1e-6                      # bss2 does not
