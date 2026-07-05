# What this project did, in plain English

## The problem: your data was chosen for you, and the AI never knew

Almost every AI is trained on data that somebody, or some process, *selected* — and
usually the AI has no idea this happened. A bank only sees whether a loan was repaid for
the loans it *granted*; it never sees what would have happened to the applicants it turned
down. A hospital only records disease outcomes for the patients it *admitted*. A team
running experiments only samples the settings someone thought were worth trying.

This matters because the piece that got left out is not random. The loan officer who
rejected an applicant often had a hunch — some scrap of information not written in any
column of the spreadsheet. So the loans you *do* see are systematically different from the
ones you don't, in a way tied to the very thing you're trying to predict. Statisticians
call this **selection on unobservables**: the invisible reason a data point was collected
is tangled up with its answer.

The standard machine-learning fix — "reweight the data you have to look like the data you
wanted" (importance weighting) — **cannot fix this**. It can correct for *who* you sampled;
it cannot correct for a bias hiding in *why* you sampled them, when that why is invisible.
You can have perfect knowledge of the sampling odds and still get the wrong answer. Worse,
a model trained this way is not just uncertain in the regions it avoided — it is
*confidently wrong* there, which is the dangerous kind.

## The 1979 idea we borrowed

Economists solved this in 1979. James Heckman won a Nobel Prize partly for a trick: instead
of pretending the missing data isn't there, you **model the selection itself**. You write
down two linked equations — one for "was this observed?" and one for "what's the outcome?"
— and let them share a hidden correlation. That correlation is exactly the invisible bias,
and once you estimate it, you can subtract it back out. The machinery has lived in
econometrics textbooks for decades. Almost nobody has plugged it into modern deep learning.
We did — twice.

## Paper A: teaching an AI to know what it doesn't know

We built the Heckman correction into a neural network's **uncertainty estimate** — its
sense of "how sure am I?" In a controlled test where we secretly know the truth, we sampled
training data in a biased way (avoiding certain regions for reasons tied to the answer) and
watched what happened:

- Ordinary confidence estimates (deep ensembles, dropout, Gaussian processes) were
  **overconfident exactly where the data had been avoided** — the worst place to be
  cocky.
- Reweighting with *perfect* knowledge of the sampling odds **did not help** — as the
  theory predicts.
- The Heckman-corrected network **restored honest uncertainty** — *but only when it had an
  "instrument"*: some extra clue that affects whether data was collected but not the answer
  itself (like which sensor took the reading). Without that clue, the fix degrades, and
  we show exactly how much rather than hiding it.

We also checked a real, slightly uncomfortable example: public AI leaderboards. Researchers
report their method on the benchmarks where it shines and quietly skip the others. That's
selection on unobservables too — and when we tried to correct it without an instrument, the
math broke down in the textbook way, which is itself the lesson: *leaderboards need a
"why did they report this" variable before anyone can de-bias them.*

## Paper B: the runs that survived were lucky, not good

Modern AI tuning uses a shortcut called **successive halving**: start hundreds of training
runs, peek early, kill the ones doing worst, keep going with the survivors. It saves
enormous compute. But there's a catch we made precise.

Early scores are noisy. A run can survive an early cut partly because it got *lucky* on a
noisy measurement, not because it's truly better. So the survivors are a mix of genuinely
good runs and lucky ones — and the lucky ones **regress to the mean**: they disappoint
later. Any tool that learns to predict final performance from these survivors inherits the
bias. It's the same Heckman/selection structure, now hiding inside a hyperparameter tuner.

We showed the bias is real and grows exactly as the theory says — with more noise and more
aggressive cutting. Then we were honest about the punchline: **at the noise levels of real
benchmark data (LCBench), the bias is tiny** — successive halving is robust where its
measurements are clean. It only bites in genuinely noisy regimes (small validation sets,
or hard tasks like ImageNet tuning, where we measured meaningful bias). We built the
selection-corrected predictor anyway; it fixes the bias in prediction, and it *rescues*
naive curve-extrapolation tuners from a "winner's curse," but it does **not** beat the
plain, cheap method at the actual tuning decisions on clean benchmarks. We report that gap
squarely instead of overselling.

## The honest thread running through both

The same caveat governs both papers, and we put it up front every time: the Heckman fix is
trustworthy **only when you have that "instrument"** — a variable that nudged whether data
was collected but not the answer. With one, it works. Without one, it leans entirely on an
assumption about the shape of the noise and gets shaky. Econometricians have known this for
forty years; we made sure a machine-learning audience can't miss it. Sometimes the useful
result is a clean "here's when it works," and sometimes it's an honest "here's a real
effect that turns out to be small in practice." This project shipped both.
