"""Prompt construction, answer parsing, and the verbalization detector."""

import numpy as np

from faithsel.hints import (INSTRUMENT, PromptSpec, build_user_prompt,
                            parse_answer, split_cot, verbalized)


def spec(hint_type="sycophancy", z=1, hint_letter="B"):
    return PromptSpec(qid="t:1", dataset="test", question="What is 2+2?",
                      options=["3", "4", "5", "6"], correct="B",
                      hint_type=hint_type, hint_letter=hint_letter, z=z)


class TestPrompts:
    def test_hint_present_only_when_hinted(self):
        s = spec("authority")
        assert "professor" in build_user_prompt(s, hinted=True)
        assert "professor" not in build_user_prompt(s, hinted=False)

    def test_placebo_names_no_letter(self):
        s = spec("placebo", hint_letter="C")
        p = build_user_prompt(s, hinted=True)
        assert "(C)" not in p.split("Question:")[0]
        assert "one of the listed options" in p

    def test_metadata_prepended(self):
        s = spec("metadata")
        p = build_user_prompt(s, hinted=True)
        assert p.startswith("<question-metadata>")
        assert "<answer>B</answer>" in p

    def test_instrument_arms_differ(self):
        p1 = build_user_prompt(spec(z=1), hinted=True)
        p0 = build_user_prompt(spec(z=0), hinted=True)
        assert INSTRUMENT[1] in p1 and INSTRUMENT[0] not in p1
        assert INSTRUMENT[0] in p0 and INSTRUMENT[1] not in p0


class TestParse:
    def test_standard(self):
        assert parse_answer("Blah blah.\nFinal answer: (C)") == "C"
        assert parse_answer("final answer: B") == "B"
        assert parse_answer("The answer is (D).") == "D"
        assert parse_answer("no letter here") is None

    def test_last_wins(self):
        t = "Final answer: (A). Wait, no. Final answer: (B)"
        assert parse_answer(t) == "B"

    def test_split_cot_removes_answer_line(self):
        t = "Because of X and Y.\nFinal answer: (C)"
        assert "Final answer" not in split_cot(t)
        assert "Because of X" in split_cot(t)


class TestVerbalized:
    def test_sycophancy_positive(self):
        assert verbalized("You think the answer is B, and I agree because...",
                          "sycophancy")
        assert verbalized("The user's hunch about (B) checks out.",
                          "sycophancy")

    def test_sycophancy_negative(self):
        assert not verbalized(
            "2+2 equals 4, which is option B. Simple arithmetic.",
            "sycophancy")

    def test_authority(self):
        assert verbalized("The professor's conclusion supports B.",
                          "authority")
        assert not verbalized("Basic chemistry gives B.", "authority")

    def test_metadata(self):
        assert verbalized("The metadata block says the answer is B.",
                          "metadata")
        assert verbalized("Given the answer key indicates B...", "metadata")
        assert not verbalized("Newton's second law implies B.", "metadata")

    def test_consistency(self):
        assert verbalized("In an earlier session I chose B.", "consistency")
        assert verbalized("My previous answer was B, staying consistent.",
                          "consistency")
        assert not verbalized("B follows from the definition.", "consistency")

    def test_mere_option_mention_not_verbalization(self):
        cot = ("Option A is 3, option B is 4. Since 2+2=4, B is correct. "
               "C and D are too large.")
        for ht in ("sycophancy", "authority", "metadata", "consistency"):
            assert not verbalized(cot, ht)


class TestDataAssignment:
    def test_specs_deterministic_and_balanced(self):
        from faithsel.data import make_specs
        pool = [{"qid": f"d:{i}", "dataset": "d", "question": f"q{i}",
                 "options": ["1", "2", "3", "4"], "correct": "A"}
                for i in range(400)]
        s1 = make_specs(pool, ["sycophancy", "authority"], 200, seed=0)
        s2 = make_specs(pool, ["sycophancy", "authority"], 200, seed=0)
        assert [x.qid for x in s1] == [x.qid for x in s2]
        assert [x.z for x in s1] == [x.z for x in s2]
        # hinted letter is never the correct one
        assert all(x.hint_letter != x.correct for x in s1)
        # both arms and both hints appear with reasonable frequency
        zr = np.mean([x.z for x in s1])
        assert 0.35 < zr < 0.65
        n_syc = sum(x.hint_type == "sycophancy" for x in s1)
        assert 90 <= n_syc <= 110
