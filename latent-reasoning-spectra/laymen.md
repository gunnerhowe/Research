# What this project found, in plain language

## The question

Some new AI systems "think" without words. Instead of writing out reasoning steps
("Sally is a scrompus. Every scrompus is a rempus..."), a model like Meta's Coconut
feeds its own internal state back into itself several times before answering — a chain
of silent, continuous "thoughts." Nobody can read those thoughts directly, which is a
problem if you care about understanding what the model is doing.

There's a tempting mathematical idea: that feedback loop is literally a *dynamical
system* — the same kind of object physicists use to describe weather, fluids, and
planetary orbits. Dynamical systems have a well-stocked toolbox (eigenvalues, stability
analysis, Koopman operators) for finding structure: where a system is about to split
between alternatives, which states steer everything downstream. If those tools worked
on thought-chains, we could *predict* where the model's reasoning branches and which
thoughts matter, straight from the math — no mind-reading required.

We tested that idea properly: hypotheses and failure conditions were written down and
committed publicly *before* any results existed, on a task (ProsQA) where the true
reasoning graph is known exactly, with control models specifically designed to expose
false positives.

## What we found

**The tempting idea fails its controls — in an unusually clear way.**

1. The spectral "signatures" that seem to mark reasoning branch points show up *just as
   strongly* in control models that provably don't have the feedback loop at all
   (they pause on a fixed token instead of recycling thoughts, and score the same on
   the task). A signature of the mechanism that survives removing the mechanism isn't a
   signature of the mechanism.

2. If you surgically rewrite a problem so there is *nothing to branch over* — deleting
   every alternative path — the supposed "branching signatures" don't change.

3. A hidden confound explained everything: early thought-steps look different from
   late ones for boring positional reasons, and reasoning branches also cluster early.
   Correct for position, and *every* signal — including the fancy ones — drops to coin-flip.

4. The most surprising discovery: in this high-performing model, the content of any
   individual silent thought barely matters. We replaced single thoughts entirely —
   with averages, even with zeros — across 3,000 trials, and the model *never once
   changed its answer*. The real work happens in the model's attention memory of the
   question; within three thought-steps, about half of a thought's influence flows
   around the feedback loop, not through it.

5. One genuinely positive nugget: what little causal influence does flow through the
   loop is concentrated in special directions the math *does* identify (the
   "non-normal amplification" directions — the same phenomenon that lets smooth water
   flow erupt into turbulence without any unstable mode). Perturbing along those
   directions has up to 7× the effect of random perturbations of the same size. The
   toolbox isn't useless — it just can't read reasoning off this system's spectrum.

## Why publishing a negative result matters

A wave of research is starting to analyze latent reasoning with exactly these
dynamical-systems tools. Without the controls we ran, that wave would produce
beautiful, structured, publishable eigenvalue plots that mean nothing — we know,
because we produced them, and then watched every one survive the removal of the thing
it supposedly measured. The paper is a pre-registered cautionary tale with the
receipts: if you want to claim spectra explain latent reasoning, here are the three
controls your claim has to survive, and here's a high-performing system where it
doesn't.

## The honest scope

This is one small model (GPT-2, 124M parameters), one synthetic logic task, one
training run. Systems trained so that thought content is genuinely load-bearing might
behave differently — and our tools transfer directly to test them.
