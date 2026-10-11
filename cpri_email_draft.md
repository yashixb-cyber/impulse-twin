# Draft — email to CPRI (Dr. Rangu Sheshu Kumar, High Voltage Division)

Not sent. Fill in the recipient address and send from your own account.

---

**To:** [Dr. Sheshu's email]
**Cc:** [rest of Team Slayers / faculty mentor, if appropriate]
**Subject:** Track 1 (Impulse Generator Settings) — two questions and a request for trial-shot data

Dear Dr. Sheshu,

We're Team Slayers, working on Track 1 (AI-driven optimisation of impulse generator
settings). Thank you for the problem statement and the Excel reference solution — both
have been central to our build. Two quick notes on where we've landed, then what we're
asking for.

**What we've built.** A circuit model of your HAEFELY 508085/E from its nameplate (12
stages, 0.125 µF/stage), solved exactly rather than approximated; IEC 60060-1 evaluation
matching the standard's procedure; your weighted k-NN reimplemented from your sheet's
formulas and checked against your reported metrics to four decimal places; and a
physics-plus-machine-learning correction that improves on it, with the honest caveat that
the gain is small on your synthetic data and larger on our own simulated hall — where the
residual has more structure to learn. Everything runs offline in a single HTML file, no
server, the way your own prototype does.

**Three questions on the hardware and standards**, so we're not guessing:

1. **Which generator is real?** The nameplate photo on page 4 of your problem statement
   reads 12 stages at 0.125 µF/stage, and we've built to that. The Excel workbook's
   calculator sheet instead models 15 stages at 3 µF/stage. Which one describes your
   actual hall?

2. **Resistors per stage.** Your nameplate lists three front-resistor values per stage
   (30 Ω / 465 Ω / 3.7 kΩ, brown/blue/yellow) and one tail value each for lightning and
   switching, each as a single line item. Does each stage carry a separate physical
   resistor for every listed value — so, say, two stages' 30 Ω resistors could be wired
   in series or parallel for a combined value — or is each stage's front position a
   single tapped unit with no combining available? We've assumed the former; if it's the
   latter, our resistor stock is wrong. Related: what do the nameplate's "connection
   possibilities" (1 p12 s; p s) mean for stage wiring, and does a parallel connection
   change the load capacitance available for switching impulses? And do the Excel
   inventory's switching-impulse settings (front 9.4–21.8 kΩ, tail ≈1.2 kΩ/stage) — which
   we can't build from either the nameplate values or the Excel inventory sheet as
   given — describe real stock, or are they illustrative?

3. **Which standard, exactly?** We're evaluating to IEC 60060-1:2010 (overshoot limit,
   switching impulse as 250/2500 µs) and IEC 60071 levels for the 420 kV class (1300 /
   1425 / 1550 kV lightning, 950 / 1050 kV switching). Are those the editions your lab
   tests against, or should we move to IEC 60060-1:2025 (170/2500 µs switching)?

**What would help most: real shot data.** Everything above is validated against a
simulator, not your hall. If you're able to share even a handful of anonymised
reduced-voltage trial shots — the settings fired and what the evaluator reported — or
IEC 61083-2 reference waveforms, that's the one thing we can't get any other way, and it
would tell us directly how our approach performs against a real generator rather than
one we built from your documents. Our correction retrains on new shots in minutes, so
we could turn results around quickly.

Happy to answer anything about the approach, and to demo the tool live if useful before
the final.

Thank you,
Team Slayers
