# The verbalization confound, in plain language

## The question

When an AI model "shows its work" — a chain-of-thought — can you trust that
the written reasoning is *why* it answered the way it did? A popular way to
check is to secretly slip a hint into the question (say, "a professor thinks
the answer is C") and see two things: (1) did the model's answer move toward
the hint, and (2) did the written reasoning *mention* the hint? If the answer
moved but the reasoning stayed silent about the hint, the reasoning is scored
"unfaithful."

## The problem with that scorecard

The reasoning you get to read is not a transcript of the model's actual
computation. It is a short, selective summary the model chose to write. And
crucially, *whether the model mentions a hint is not independent of whether it
relied on it.* Maybe it tends to mention hints exactly when it leaned on them —
or exactly when it didn't. Either way, counting "did it mention the hint?" over
only the cases where it *did* mention something gives you a biased picture. You
are measuring a self-selected sample.

Economists have known about this exact trap since 1979. If you only observe
wages for people who *chose* to work, you can't read off the population wage
from the workers alone — the choice to work is tangled up with the wage. James
Heckman won a Nobel Prize partly for a fix: model the *choice to be observed*
and the *thing you care about* as two linked equations, and correct for how
they're correlated. That correlation has a name, ρ (rho).

## What this project does

We are the first to point out that AI chain-of-thought faithfulness is the very
same statistical problem, and to import the fix. We build:

- **A test.** Is ρ different from zero? If yes, "counting verbalizations" is
  genuinely biased and needs correcting. If no, the simple count was fine.
- **A correction.** A formula that turns the biased count into an unbiased
  estimate — including an estimate of how much the model *relied on the hint in
  the cases where it never mentioned it*, the reliance the simple count silently
  records as zero.
- **A deployment path.** The correction needs only what you can see from the
  outside (the text and the answer), so it works on closed models like Claude
  where you can't inspect the internals.

We check the whole thing on open models where we *can* peek inside (using a
clean causal test: delete the hint and see how much the answer's internal
"vote" shifts). That gives us the true reliance for every case, so we can
verify the correction recovers it — then we apply the same correction to a
closed API model where the truth is hidden.

## What we actually found (an honest negative)

We set out to show the correction works. It mostly doesn't — and *why* it
doesn't turned out to be the more useful result.

1. **The standard way of measuring "did the reasoning mention the hint?" is
   badly wrong.** The field uses a keyword detector. When we checked it against
   a careful reader (another AI model reading each trace and judging), the
   detector over-counted "mentions" by **2× to 9×** — it fired on innocent
   words like "the *source* of gravity" or "as noted *earlier*." Once you
   measure mentions correctly, they're **rare** (roughly 3–25% of the time),
   matching what other labs have found. Since faithfulness scores are built on
   this measurement, getting it wrong is a bigger problem than any correction
   can fix.

2. **When we measure mentions correctly, the naive faithfulness number is
   already about right**, and our fancy correction makes it *worse*, not
   better — it overshoots. The statistical test still flags that "selection"
   is happening, but a sensitivity check shows the correction that would
   actually match reality is essentially "do nothing." The overshoot comes
   from an assumption (bell-curve-shaped errors) that the real data violates.

3. So the honest bottom line: the *idea* (faithfulness metrics are a
   self-selected sample) is right and the *test* is a useful warning light,
   but the *correction* doesn't deliver on these models, and the thing
   practitioners should worry about most is measuring "verbalization"
   reliably in the first place.

This is a negative result, reported plainly. Negative results that explain
*why* a natural idea fails — and surface a measurement problem the field
should fix — are worth writing down.
