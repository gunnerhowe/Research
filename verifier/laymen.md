# Do AI research judges reward good ideas — or ideas that *sound* good?

**Plain-language summary.**

## The question

People increasingly want to use large language models (LLMs) as automatic
*judges* — to score research ideas, rank AI-generated hypotheses, even to be the
"reward" that trains the next AI to do research. For that to work, the judge has
to reward ideas that are actually *novel and good*, not ideas that merely *sound*
impressive. We asked: which one do these judges actually do?

## How we tested it

We wrote 154 research "stems." For each one we made four versions in a clean 2×2:

- an **incremental** idea vs. a **genuinely novel** idea (real difference in
  substance — checked by an independent AI reviewer that never saw the framing), and
- each written **plainly** vs. dressed in **novelty rhetoric** ("contrary to the
  prevailing view… a fundamentally new… surprising… unprecedented") — with the
  *exact same technical content*, only the wording changed.

Then three open LLM judges scored each version's novelty, blind. Because the only
thing that changes between "plain" and "dressed-up" is the rhetoric, any score
difference is the judge reacting to *hype*, not to the idea.

## What we found

1. **The judges reward the rhetoric, not the substance.** Across all three
   judges, the boost from novelty-sounding *wording* was larger than the boost
   from actually being *more novel*. Concretely: **~82% of the time, a boring idea
   dressed in novelty language beat a genuinely novel idea stated plainly.**

2. **It isn't just "longer text."** The dressed-up versions were only ~6% longer,
   and when we statistically control for length the effect doesn't shrink at all.

3. **It holds in head-to-head comparison** (for two of the three judges), so it's
   a real shift in judgment, not a quirk of how we read the score.

4. **The "hype signal" is a clear switch inside the model.** We can read, from the
   judge's internal activations, whether it just saw novelty rhetoric — with
   near-perfect accuracy, and it even transfers from machine-learning ideas to
   economics ideas.

5. **But you can't just turn the switch off.** The obvious fix — surgically
   subtracting that "hype direction" from the model's activations — **does not
   work.** It either does nothing, or it damages the judge's ability to tell good
   from bad *along with* the bias. The hype-reaction is woven into how the judge
   thinks about novelty, not a separate dial you can remove. (For other known
   judge biases, that same subtraction trick *does* work — so this one is
   unusually deeply baked in.)

6. **One simple story explains two puzzling prior results.** Two 2026 benchmarks
   disagreed: one found judges *over*-rate AI-written ideas, another found their
   novelty scores are compressed and detached from expert judgment. If judges
   score *hype* instead of *substance*, you get **both** at once — hype-heavy
   ideas float up, plainly-stated real novelty sinks, and the score stops tracking
   real novelty.

## Why it matters

If we wire an LLM judge into a loop that trains AI to "do novel research," this
says the AI will learn to *sound* novel — to generate confident, paradigm-shifting
*language* — rather than to actually be novel. And because the bias can't be
cheaply patched out, that's not a small bug to fix later; it's a reason to be
cautious about using today's LLM judges as the verifier for automated research.

## What this is *not* (honest limits)

- Small, open judges (7–8B parameters) and *synthetic* ideas — not (yet) frontier
  judges or a real peer-review corpus.
- The effect's size depends on how you read the score; for one judge it shrank to
  borderline in head-to-head comparison.
- "Near-perfect detection" of the hype signal is partly *expected* — the rhetoric
  words are right there in the text — so the interesting part is the causal test
  (can't remove it), not the detection itself.

*Part of the `verifier` project: is the missing piece for automated science a
better generator, or a better judge? Here, the judge is the problem.*
