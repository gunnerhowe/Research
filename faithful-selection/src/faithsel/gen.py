"""GPU engine: 4-bit model loading, batched greedy CoT generation, and the
forward-pass measurements behind the ground-truth reliance variables.

Per instance we obtain (all greedy / deterministic):
- gen_hinted, gen_unhinted: full CoT + answer under the hinted prompt and the
  matched no-hint control (same instrument arm).
- Three "letter reads": next-token log-probs over {A,B,C,D} after
  <prompt> + <CoT> + "Final answer: (" for
    (hinted prompt, hinted CoT)      -> total-effect endpoint
    (unhinted prompt, unhinted CoT)  -> total-effect baseline
    (unhinted prompt, hinted CoT)    -> hint-excision splice (NDE-style:
                                        same CoT, hint removed from prompt)
- Logit-lens read: per-layer letter distributions at "Final answer: (" with
  NO CoT, for hinted and unhinted prompts (instrument sentence excluded by
  construction), giving the pre-verbalization commitment measure and the
  difficulty covariate (unhinted direct-answer entropy / correctness).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .hints import LETTERS, SYSTEM_PROMPT, PromptSpec, build_user_prompt, split_cot

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "nemotron8b": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
}

ANSWER_SUFFIX = "\nFinal answer: ("


def load_model(key: str):
    name = MODELS[key]
    qcfg = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=qcfg, device_map="cuda:0",
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def chat_text(tok, user: str, system: str = SYSTEM_PROMPT) -> str:
    """Rendered chat-template text up to and including the generation prompt."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def letter_token_ids(tok) -> list[int]:
    ids = []
    for letter in LETTERS:
        enc = tok.encode(letter, add_special_tokens=False)
        assert len(enc) == 1, f"letter {letter} not a single token: {enc}"
        ids.append(enc[0])
    return ids


@torch.no_grad()
def generate_batch(model, tok, texts: list[str],
                   max_new_tokens: int = 448) -> list[str]:
    """Greedy batched generation from pre-rendered chat texts."""
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


@torch.no_grad()
def read_letters_batch(model, tok, texts: list[str]) -> np.ndarray:
    """Full-vocab log-softmax at the last position, restricted to the four
    letter tokens. Returns (B, 4) letter log-probs."""
    lids = letter_token_ids(tok)
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[:, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    return logp[:, lids].cpu().numpy()


def _final_norm(model):
    for path in ("model.norm", "model.final_layernorm", "transformer.ln_f"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate final norm module")


@torch.no_grad()
def lens_letters_batch(model, tok, texts: list[str]) -> np.ndarray:
    """Logit-lens letter distributions per layer at the last position.

    Returns (B, n_layers+1, 4) RESTRICTED letter probabilities (softmax over
    the four letter logits) after applying the final norm + unembedding to
    each layer's residual stream.
    """
    lids = letter_token_ids(tok)
    norm = _final_norm(model)
    head = model.get_output_embeddings()
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    outs = []
    for h in hs:                       # embeddings + each layer
        v = norm(h[:, -1, :])
        logits = head(v).float()[:, lids]
        outs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.stack(outs, dim=1).numpy()


# ------------------------------------------------------------ per-chunk run


def restricted_logprob(letter_logps: np.ndarray, idx: int) -> float:
    """log p(letter idx) after renormalizing over the four letters."""
    x = np.asarray(letter_logps, dtype=float)
    x = x - np.logaddexp.reduce(x)
    return float(x[idx])


def process_chunk(model, tok, specs: list[PromptSpec],
                  max_new_tokens: int = 448, gen_bs: int = 8,
                  read_bs: int = 8, lens_bs: int = 4,
                  do_lens: bool = True) -> list[dict]:
    """Run the full measurement pipeline for a chunk of instances."""
    text_h = [chat_text(tok, build_user_prompt(s, hinted=True)) for s in specs]
    text_u = [chat_text(tok, build_user_prompt(s, hinted=False)) for s in specs]

    def batched(fn, texts, bs, **kw):
        out = []
        for i in range(0, len(texts), bs):
            out.extend(fn(model, tok, texts[i:i + bs], **kw))
        return out

    gen_h = batched(generate_batch, text_h, gen_bs,
                    max_new_tokens=max_new_tokens)
    gen_u = batched(generate_batch, text_u, gen_bs,
                    max_new_tokens=max_new_tokens)

    cot_h = [split_cot(t).strip() for t in gen_h]
    cot_u = [split_cot(t).strip() for t in gen_u]

    read_hh = [th + ch + ANSWER_SUFFIX for th, ch in zip(text_h, cot_h)]
    read_uu = [tu + cu + ANSWER_SUFFIX for tu, cu in zip(text_u, cot_u)]
    read_uh = [tu + ch + ANSWER_SUFFIX for tu, ch in zip(text_u, cot_h)]

    lp_hh = np.stack(batched(read_letters_batch, read_hh, read_bs))
    lp_uu = np.stack(batched(read_letters_batch, read_uu, read_bs))
    lp_uh = np.stack(batched(read_letters_batch, read_uh, read_bs))

    lens_h = lens_u = None
    if do_lens:
        # lens prompts exclude the instrument sentence by construction
        def z_free(s: PromptSpec, hinted: bool) -> str:
            import dataclasses
            s0 = dataclasses.replace(s)
            up = build_user_prompt(s0, hinted=hinted)
            from .hints import INSTRUMENT
            up = up.replace("\n\n" + INSTRUMENT[s.z], "")
            return chat_text(tok, up) + ANSWER_SUFFIX

        lens_texts_h = [z_free(s, True) for s in specs]
        lens_texts_u = [z_free(s, False) for s in specs]
        lens_h = np.concatenate(
            [lens_letters_batch(model, tok, lens_texts_h[i:i + lens_bs])
             for i in range(0, len(specs), lens_bs)])
        lens_u = np.concatenate(
            [lens_letters_batch(model, tok, lens_texts_u[i:i + lens_bs])
             for i in range(0, len(specs), lens_bs)])

    rows = []
    for j, s in enumerate(specs):
        row = {
            "qid": s.qid, "dataset": s.dataset, "hint_type": s.hint_type,
            "hint_letter": s.hint_letter, "correct": s.correct, "z": s.z,
            "question_len": len(s.question),
            "gen_hinted": gen_h[j], "gen_unhinted": gen_u[j],
            "lp_hh": lp_hh[j].tolist(), "lp_uu": lp_uu[j].tolist(),
            "lp_uh": lp_uh[j].tolist(),
        }
        if do_lens:
            row["lens_h"] = np.round(lens_h[j], 5).tolist()
            row["lens_u"] = np.round(lens_u[j], 5).tolist()
        rows.append(row)
    return rows
