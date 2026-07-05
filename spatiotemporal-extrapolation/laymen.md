# In plain terms

## The question

Some physical systems are chaotic *and* extended in space — think of a turbulent
fluid, a flame front, or ripples on a long channel. You cannot predict their exact
future (chaos erases that), but you *can* predict their **statistics**: how quickly
patterns lose memory of themselves, what the typical wavelengths are, how long
correlations last. Those statistics are encoded in the "spectrum" of a
mathematical object called the transfer operator.

Simulating or learning such a system on a **big** domain is expensive. So we asked:
can we learn the operator's spectrum on a few **small** domains, watch how it
changes as the domain grows, and **extrapolate** to a domain 64 times larger —
without ever simulating the big one?

We tested this on the Kuramoto–Sivashinsky equation, a standard 1-D model of
spatiotemporal chaos, on a single gaming GPU.

## What we found

**Good news first.** The spectrum *does* change smoothly and predictably as the
domain grows (we call this passing "Gate S"). Fitting that smooth trend on domains
of size 22–88 and extrapolating to size 1408 (64× the volume) predicts the big
domain's decay rates to about **10%**, its power spectrum to about **12%**, and its
correlation function to **7%** — with zero data from the big domain. The method
works.

**The honest catch.** We also ran the dumbest possible baseline: just take the
spectrum of the **largest small box you can afford** (size 88) and stretch it onto
the big grid, with *no* clever scaling at all. That baseline does **better** (3%
vs. 10%). Why? Because this particular system settles into its large-size behavior
so quickly — by size 88 it's already within a few percent of the size-1408 answer —
that there's nothing left for a scaling law to correct. We pre-registered this as
a possible outcome ("kill condition K2"), and it happened. We report it plainly.

This is the point of the paper. A good null control is what separates a real result
from wishful thinking. Our scaling machinery is sound and it works; it's just
**unnecessary** for a system that converges this fast. We spell out exactly where
it *would* be needed: systems whose "memory length" is comparable to the biggest
box you can simulate, or where growing the domain creates genuinely new large-scale
instabilities — there, stretching the small box must fail, and only a scaling law
can bridge the gap.

**A side finding.** We also trained a deep neural network version of the operator.
It faithfully reproduces the system's *statistics* (the power spectrum, to ~8%) but
gets the *individual vibration frequencies* of each mode badly wrong. So we built
the scaling law on the classical, reliable estimator instead — which is itself a
"learned" operator, just not a neural one.

## Why it's still worth publishing

Three things are new and solid: (1) the first measurement that these learned
operator spectra scale smoothly with domain size; (2) a working — if, here,
redundant — recipe to extrapolate them 64× with no big-domain data; and (3) a
clean, quantified demonstration of *when the simple baseline wins and why*, with a
concrete prediction of where it must lose. Every number in the paper is generated
by code from raw simulation output and regenerates identically on re-run.
