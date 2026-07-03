# arXiv endorsement request emails

**Before sending:** register at arxiv.org, start a submission to **cs.LG**, and
arXiv will show your personal endorsement code and link (looks like
`https://arxiv.org/auth/endorse?x=XXXXXX`). Paste it where marked
`[ENDORSEMENT LINK + CODE]`. Attach `paper/main.pdf` to each email.

Verified contacts (all taken from the authors' own published papers):

| Person | Email | Source | Why them |
|---|---|---|---|
| Ali Veisi | ali.veisi@axiomlab.org | CABLE paper, title page | Their method is benchmarked in our paper |
| Hamidreza Amirzadeh | h.amirzadeh@axiomlab.org | CABLE paper, title page | ditto |
| Amir Mansourian | amir.mansurian@axiomlab.org | CABLE paper, title page (note spelling "mansurian") | ditto |
| Simran Arora | simran@cs.stanford.edu | Zoology paper, title page | Our diagnostics build on Zoology/MQAR |
| Ofir Press | ofirp@cs.washington.edu | ALiBi paper, title page — **2021 address, may be stale; he has since moved (see ofir.io)** | SemRF strictly generalizes ALiBi |
| Luca Moschella | *no public email* — use the contact form at luca.moschella.dev (now at Apple) | personal site | Relative representations inspired SemRF |

---

## Email 1 — CABLE authors (send to all three; best odds of reply)

**To:** ali.veisi@axiomlab.org, h.amirzadeh@axiomlab.org, amir.mansurian@axiomlab.org
**Subject:** arXiv endorsement request (cs.LG) — paper benchmarking CABLE

Dear Ali, Hamidreza, and Amir,

I'm an independent researcher and I've just finished a paper that builds
directly on your work: "Semantic Reference Frames: Representing Time and
Context Relative to Learned Anchors in Sequence Models." It proposes a
structured content-conditioned attention bias (anchor frames + frame-dependent
temporal decay) and includes CABLE as a primary baseline — implemented
faithfully from your reference repository — across enwik8 and three synthetic
diagnostics under matched backbones and compute.

You may find the results interesting: CABLE's cumulative-sum design wins
decisively on associative recall with long uninformative gaps (perfect
accuracy at every gap length — the cleanest result of any method there), while
our frame-structured variant leads on enwik8 bits-per-character and on an
ordered-copying task. I've attached the PDF; your work is discussed and
compared throughout.

As this is my first arXiv submission, I need an endorsement for cs.LG. If,
after looking at the paper, you'd be comfortable endorsing, here is my
endorsement link and code:

[ENDORSEMENT LINK + CODE]

If not, no worries at all — and I'd welcome any comments on the comparison
either way.

Best regards,
Gunner Howe
gunnerlevihowe@gmail.com

---

## Email 2 — Simran Arora (Zoology)

**To:** simran@cs.stanford.edu
**Subject:** arXiv endorsement request (cs.LG) — diagnostics built on Zoology/MQAR

Dear Simran,

I'm an independent researcher seeking an arXiv endorsement for cs.LG for my
first submission: "Semantic Reference Frames: Representing Time and Context
Relative to Learned Anchors in Sequence Models" (PDF attached).

The paper proposes a content-conditioned attention bias in which tokens are
softly assigned to learned semantic anchors and the temporal decay rate is a
learned function of the anchor frame. The experimental design leans directly
on your Zoology work: our associative-recall diagnostics are MQAR-style, and
we independently ran into (and document) the recall-capacity and circuit-
formation effects your paper describes — including a sharp, budget-dependent
phase transition in induction-circuit formation that made careful task
calibration essential. Zoology is cited in the setup, results, and appendix.

If, after a look at the paper, you'd be comfortable endorsing, here is my
endorsement link and code:

[ENDORSEMENT LINK + CODE]

Absolutely no problem if not — and thank you for Zoology; it saved this
project from an uncalibrated comparison.

Best regards,
Gunner Howe
gunnerlevihowe@gmail.com

---

## Email 3 — Ofir Press (ALiBi)

**To:** ofirp@cs.washington.edu  *(from the ALiBi paper; if it bounces, he lists
current contact links at ofir.io)*
**Subject:** arXiv endorsement request (cs.LG) — a strict generalization of ALiBi

Dear Ofir,

I'm an independent researcher seeking an arXiv endorsement for cs.LG for my
first submission: "Semantic Reference Frames: Representing Time and Context
Relative to Learned Anchors in Sequence Models" (PDF attached).

The method is, by construction, a strict generalization of ALiBi: tokens are
softly assigned to a small set of learned semantic anchors, and each anchor
frame learns its own per-head decay slope — with the model initialized exactly
at ALiBi's operating point (your published slopes) and the content terms gated
near zero. On enwik8 the frame-conditioned decay inherits ALiBi's
extrapolation behaviour (monotonically improving bpc out to 8x the training
context) while improving bits-per-character at every length; the trained
frames turn out to be interpretable, with structural tokens learning slow
decay and word-internal letters fast decay. ALiBi is, unsurprisingly, the
paper's central baseline and reference point.

If you'd be comfortable endorsing after a look, here is my endorsement link
and code:

[ENDORSEMENT LINK + CODE]

No problem at all if not — and thanks for ALiBi, which this work quite
literally starts from.

Best regards,
Gunner Howe
gunnerlevihowe@gmail.com

---

## Contact-form message — Luca Moschella (relative representations)

*(No public email; paste into the contact form at luca.moschella.dev.)*

Subject: arXiv endorsement request — relative representations applied to position/time

Dear Luca,

I'm an independent researcher and my first arXiv submission, "Semantic
Reference Frames: Representing Time and Context Relative to Learned Anchors in
Sequence Models," transfers the anchor-relative representation principle from
your ICLR 2023 paper to a new object: positional/temporal information in
transformer attention. Tokens are softly assigned to learned anchors and the
attention bias — including a learned per-frame temporal decay — is built
frame-relatively rather than from absolute indices. Your paper is credited in
the introduction and related work as the origin of the principle.

I need an endorsement for cs.LG. If you'd be comfortable endorsing after a
look at the paper (I can send the PDF to any address you prefer), here is my
endorsement link and code:

[ENDORSEMENT LINK + CODE]

No worries at all if not — either way, thank you for the relative
representations work; it was the seed of this project.

Best regards,
Gunner Howe
gunnerlevihowe@gmail.com

---

### Sending tips
- Endorsers must be established authors **in cs.LG specifically**; all of the
  above qualify.
- Send the CABLE email first (highest response odds), then Arora, then Press.
  One endorsement is enough — you can stop once someone accepts.
- Keep the PDF attached; endorsers are asked to judge whether the work is
  plausibly of archival interest, so making it one click away matters.
- If nothing lands within ~a week, arXiv's help pages also describe finding
  endorsers via the "Which of these authors can endorse?" link on any recent
  cs.LG paper.
