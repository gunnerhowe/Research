"""End-to-end analysis pipeline on synthetic raw rows: a fake 'model' whose
verbalization is endogenously selected on latent reliance (rho > 0) by
construction. fit_report must detect the confound and correct it."""

import numpy as np
import pandas as pd
import pytest

from faithsel.analysis import augment, design, e1_balance, fit_report
from faithsel.hints import LETTERS


def synth_rows(n=1500, rho=0.7, seed=0):
    """Raw JSONL-like rows. Latent reliance R drives both the letter
    log-probs and (through a correlated error) whether the fake CoT
    mentions the hint."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        h = int(rng.integers(0, 4))
        hint_letter = LETTERS[h]
        correct = LETTERS[(h + 1) % 4]
        z = int(rng.integers(0, 2))
        qlen = float(rng.integers(40, 400))
        ds = ["mmlu", "arc_easy", "obqa"][i % 3]
        ht = ["sycophancy", "authority"][i % 2]

        err = rng.multivariate_normal([0, 0],
                                      [[1.0, rho], [rho, 1.0]])
        R = 0.8 + 0.3 * (ht == "authority") + 1.0 * err[1]
        v_lat = -0.4 + 1.2 * z + 0.2 * (ht == "authority") + err[0]
        V = v_lat > 0

        # letter log-probs realizing R_TE = R exactly (restricted logits)
        base = np.full(4, np.log(0.25))
        lp_uu = base.copy()
        p_h_hinted = 1.0 / (1.0 + np.exp(-(np.log(1 / 3.) + R)))
        lp_hh = np.log(np.r_[p_h_hinted,
                             [(1 - p_h_hinted) / 3.0] * 3])
        lp_hh = np.roll(lp_hh, h)
        lp_uh = lp_hh * 0.5 + lp_uu * 0.5

        cues = {"sycophancy": "You think it is ({}) and that matters. ",
                "authority": "The professor concluded ({}), worth weighing. "}
        cot = (cues[ht].format(hint_letter)
               if V else "The definition implies the result. ")
        ans = hint_letter if R > 1.0 else correct
        gen_h = cot + f"Final answer: ({ans})"
        gen_u = "Standard reasoning. Final answer: ({})".format(correct)
        rows.append({
            "qid": f"s:{i}", "dataset": ds, "hint_type": ht,
            "hint_letter": hint_letter, "correct": correct, "z": z,
            "question_len": qlen, "gen_hinted": gen_h, "gen_unhinted": gen_u,
            "lp_hh": lp_hh.tolist(), "lp_uu": lp_uu.tolist(),
            "lp_uh": lp_uh.tolist(),
        })
    return pd.DataFrame(rows)


class TestAugment:
    def test_variables(self):
        df = augment(synth_rows(200))
        assert set(["V", "R_TE", "R_NDE", "followed", "flip",
                    "parse_ok"]).issubset(df.columns)
        assert df["parse_ok"].all()
        # R_TE reproduces the construction: hinted restricted-logit shift
        assert df["R_TE"].std() > 0.5

    def test_verbalization_tracks_construction(self):
        df = augment(synth_rows(500))
        # V=1 rows were built with a hint-type-matched cue
        v1 = df.loc[df["V"] == 1, "gen_hinted"]
        assert v1.str.contains("You think|The professor", regex=True).all()
        v0 = df.loc[df["V"] == 0, "gen_hinted"]
        assert not v0.str.contains("You think|The professor",
                                   regex=True).any()


class TestFitReport:
    @pytest.fixture(scope="class")
    def report(self):
        df = augment(synth_rows(2500, rho=0.7, seed=1))
        return fit_report(df, outcome="R_TE", n_boot=60, per_hint=False)

    def test_detects_confound(self, report):
        assert report["rho_lr"]["p"] < 0.01
        assert report["mle"]["rho"] > 0.3

    def test_correction_beats_naive(self, report):
        g = report["gate"]
        assert g["corrected_beats_naive_selected"]
        assert g["corrected_beats_naive_hidden"]

    def test_first_stage_strong(self, report):
        assert report["gamma_z_first_stage"]["z_stat"] > 4

    def test_null_case_no_detection(self):
        df = augment(synth_rows(2500, rho=0.0, seed=2))
        rep = fit_report(df, outcome="R_TE", n_boot=0, per_hint=False,
                         do_sensitivity=False)
        assert rep["rho_lr"]["p"] > 0.01
        # without a confound the naive selected mean is nearly unbiased
        assert abs(rep["two_step"]["estimands"]["naive_selected"]
                   - rep["targets"]["true_pop"]) < 0.15


class TestBalance:
    def test_e1_shape(self):
        df = augment(synth_rows(1000))
        out = e1_balance(df, outcomes=("R_TE", "R_NDE"))
        assert out["V_rate_z1"] > out["V_rate_z0"]     # instrument moves V
        assert out["first_stage"]["p"] < 0.01
        # Z was not wired into R in the construction: balance holds
        assert abs(out["R_TE"]["std_diff"]) < 0.15


class TestDesign:
    def test_shapes_and_exclusion(self):
        df = augment(synth_rows(300))
        X, W, names = design(df)
        assert W.shape[1] == X.shape[1] + 1
        assert names[-1] == "z"
        assert X.shape[0] == len(df)
