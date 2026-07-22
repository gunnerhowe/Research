"""Local 4-bit judge engine: load, score, and (for E3/E4) read + steer activations.

Score readout is the *expected digit* over the next-token distribution, not a
decoded string: after the chat template's generation prompt, we softmax the
model's next-token logits restricted to the tokens for digits 1-9 and take the
expectation. This is deterministic, parse-failure-free, and lower-variance than
sampling — the right Y for a mixed-effects fit.

Hooks:
  - `capture_residual(model, layer)` — grab the residual stream at the scoring
    position (last prompt token) for probing the signal direction (E3).
  - `steer(model, layer, vector, coeff)` — add a direction to the residual stream
    during scoring, to test the calibration fix (E4).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from novjudge.rubric import SCALE_MIN, SCALE_MAX


@dataclass
class Judge:
    model_id: str
    model: object
    tok: object
    digit_ids: dict          # digit -> list[int] candidate token ids
    device: str = "cuda"


def load_judge(model_id: str, quantize: bool = True) -> Judge:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.float16, device_map={"": 0})
    if quantize:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    digit_ids = {}
    for d in range(SCALE_MIN, SCALE_MAX + 1):
        ids = set()
        for form in (str(d), " " + str(d)):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[-1])
        digit_ids[d] = sorted(ids)
    return Judge(model_id=model_id, model=model, tok=tok, digit_ids=digit_ids)


def _prompt_ids(judge: Judge, messages: list[dict]) -> torch.Tensor:
    ids = judge.tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    return ids.to(judge.model.device)


@torch.no_grad()
def expected_score(judge: Judge, messages: list[dict]) -> float:
    """E[digit] under the next-token distribution restricted to digits 1-9."""
    ids = _prompt_ids(judge, messages)
    logits = judge.model(ids).logits[0, -1, :].float()
    digits = list(range(SCALE_MIN, SCALE_MAX + 1))
    logit_per_digit = torch.tensor(
        [torch.logsumexp(logits[judge.digit_ids[d]], dim=0) if judge.digit_ids[d]
         else torch.tensor(-1e30) for d in digits]
    )
    p = torch.softmax(logit_per_digit, dim=0)
    return float(sum(d * pi for d, pi in zip(digits, p.tolist())))


def _layers(model):
    # Llama/Qwen/Phi/Nemotron all expose decoder blocks here.
    return model.model.layers


@contextlib.contextmanager
def capture_residual(judge: Judge, layer: int):
    """Yield a dict; after a forward pass, dict['h'] holds the residual stream at
    the last (scoring) position for the given decoder layer output."""
    store = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h[:, -1, :].detach().to(torch.float32).cpu()
        return out

    handle = _layers(judge.model)[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@contextlib.contextmanager
def steer(judge: Judge, layer: int, vector: torch.Tensor, coeff: float):
    """Add coeff * vector to the residual stream at `layer` for every position
    during the forward pass (activation steering for the E4 fix)."""
    vec = vector.to(judge.model.device, dtype=torch.float16)

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + coeff * vec
            return (h,) + tuple(out[1:])
        return out + coeff * vec

    handle = _layers(judge.model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def score_and_capture(judge: Judge, messages: list[dict], layer: int) -> tuple[float, torch.Tensor]:
    """One forward pass returning both the expected score and the residual at
    `layer` (for building probes / difference-of-means directions)."""
    with capture_residual(judge, layer) as store:
        y = expected_score(judge, messages)
    return y, store["h"][0]
