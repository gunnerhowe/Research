"""Offline mode: watch a public checkpoint stream (the paper's Pythia/OLMo protocol).

Probes each revision of a HuggingFace checkpoint suite, feeds the same anchor/event
machinery as the live callback, and prints the alarm -> forecast -> event sequence.
Run on CPU or GPU; downloads checkpoints on demand.

    python examples/watch_checkpoint_stream.py --model EleutherAI/pythia-70m \
        --revisions step256,step512,step1000,step2000
"""
import argparse
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM

try:
    import emergence_watch  # noqa: F401  (installed via pip)
except ModuleNotFoundError:  # running from a repo checkout
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emergence_watch import (ComposedAnchor, Signal, paper_induction_event,
                             make_induction_probe_fn, LiveMonitor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--revisions", required=True, help="comma-separated revision names")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # single-gate anchor here because public streams carry no behavioral-ramp probe;
    # for your own training runs use the composed anchor (see README table).
    anchor = ComposedAnchor([Signal("prevtok_early", 0.10)])
    monitor = LiveMonitor(probe_fn=None, anchor=anchor, event=paper_induction_event(),
                          cadence=1, jsonl_path="checkpoint_stream.jsonl")

    probe_fn = None
    for i, rev in enumerate(args.revisions.split(",")):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=rev, torch_dtype=torch.float32,
            attn_implementation="eager").to(args.device).eval()
        if probe_fn is None:
            probe_fn = make_induction_probe_fn(model.config.vocab_size,
                                               device=args.device)
            monitor.probe_fn = probe_fn
        rec = monitor.step(model, i, force=True)
        print(rev, {k: round(v, 4) for k, v in rec.items()
                    if isinstance(v, float)}, "ALARM" if rec["anchor_fired"] else "")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(monitor.summary())


if __name__ == "__main__":
    main()
