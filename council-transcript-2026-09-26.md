# LLM Council — Impulse Twin, PowerNext-AI Track 1

Run: 2026-09-26. Question framed for the council, all five independent advisor responses, the anonymization mapping, all five peer reviews, and the chairman synthesis.

## Original question

Is Team Slayers' impulse-generator decision-support project ("Impulse Twin," PowerNext-AI Track 1) likely to win the hackathon final, and how should the team spend its last two weeks?

## Framed question put to the council

Team Slayers is building "Impulse Generator Settings — Decision Support" for PowerNext-AI Track 1, a hackathon run by CPRI with MIT Bengaluru, 24-hour final 10-11 October 2026, ~2 weeks away. Judges are CPRI High Voltage Division engineers. Judged explicitly on accuracy/precision/MAPE/MSE/RMSE and which algorithm/tuning performs best, framed as a regression problem, against CPRI's own weighted-kNN reference solution. The team built exact circuit physics, deterministic IEC waveform evaluation, CPRI's kNN reproduced to 4 decimals, then a hybrid ML correction, a browser-deployable cross-test "memory," overshoot prediction, atmospheric correction, similar-shots evidence — all in one offline HTML app with a live shot-loop. On a SIMULATED hall they built themselves, their "lab memory" strategy beats a simulated technician baseline at p<0.0001 (shots) and p=0.0007 (climbs), surviving Bonferroni correction; a browser-only version also beats it (p=0.0001/0.0004) and beats itself-without-memory (p=0.0003). On CPRI's OWN real 2000-row dataset, they beat CPRI's kNN narrowly (4-6% relative MAPE), not surviving correction across all comparisons. No real CPRI trial-shot data exists yet. Which of CPRI's two hardware descriptions is the real generator (12-stage/0.125µF nameplate vs 15-stage/3µF Excel calculator) is unconfirmed. A CPRI-data-request email is drafted but unsent. One tried AI approach (black-box Bayesian optimization) is worse than the baseline, reported honestly. Question: how good a shot does this have at winning, what's the biggest unseen risk, what's the highest-leverage move in 2 weeks?

## Advisor responses

### The Contrarian

The single biggest risk this team isn't seeing: they've built a research paper when the rubric asks for a working demo of a device that changes resistors on a real generator.

Look at what "beats CPRI's kNN" actually means: 4-6% relative MAPE gain on THEIR data, p-values that die the moment you correct across 15 comparisons. That's not a hackathon-winning result, that's a null result dressed up in confidence intervals. The judges are CPRI engineers who run impulse halls — they will read "explains only 40% of the front-time residual, the rest is noise" and correctly conclude the ML layer is decoration on top of physics that was already good. If the entire pitch rests on "our kNN beats your kNon-kNN," ten teams are probably converging on variations of the same reference solution CPRI handed everyone. Marginal beats of marginal beats is not a differentiator in a room full of teams doing the same homework.

The bigger tell: the "big win" (p<0.0001 lab memory result) lives entirely inside a simulator this team built, calibrated against no real hall. That's not evidence, it's a model agreeing with itself's cousin — the team already knows this (rule #2 in their own CLAUDE.md), but the headline result of the whole project is exactly this kind of self-referential proof. A CPRI engineer's first question will be "what happens on MY generator," and the honest answer is "we don't know, we've never touched real data."

Highest-leverage move for two weeks: stop tuning the simulator and stop chasing another 0.5% MAPE. Send the CPRI email TODAY, not as an afterthought — every day it sits in drafts is a day where the single dataset that could validate or falsify everything doesn't exist. And build the failure story into the pitch on purpose: show the black-box BO loss, show the 15-stage-vs-nameplate hardware uncertainty, show the ceiling-is-noise finding on their own data — because a room of skeptical engineers trusts a team that shows its own model's blind spots more than one that only shows wins. Right now the deck's strongest asset (intellectual honesty) and its biggest exposure (zero real validation) are the same fact, and nothing in the plan addresses the second while there's still time to.

### The First Principles Thinker

Strip this down to what's actually being judged, not what you've built. CPRI's rubric says: accuracy, precision, MAPE, MSE, RMSE, against a regression problem, on THEIR data and THEIR baseline. Everything else — lab memory, app memory, the shot-loop, overshoot prediction, the whole simulated-hall benchmark — is a second project you built on top, validated against a world you invented. It's impressive, but it isn't what's on the rubric. If a judge asks "what's your MAPE against our kNN on our validation set," the honest answer is 4-6%, borderline after correction. That's the actual scoreboard. Don't let the bigger, shinier simulator story upstage the small, real number — a CPRI engineer will ask for that number first.

The single biggest risk nobody's naming: you don't know which generator you're modeling. Nameplate says 12 stages/0.125µF; CPRI's own Excel says 15 stages/3µF. Every circuit result, every resistor recommendation, every "climbs saved" number is downstream of that choice. If the jury's own generator is the 15-stage one and you built for 12, a domain-expert judge catches this in the first five minutes, and everything else — no matter how well-benchmarked — is now "impressive work on the wrong machine." That single fact, not model choice, not p-values, is what can zero out two weeks of work in front of these particular judges.

Highest-leverage move with 2 weeks left: send the CPRI email today, not later. Not as due diligence — as the literal precondition for whether Track 1 of this project points at reality. Everything else is well-optimized speculation until that's answered.

### The Expansionist

Building a case for what CPRI's biggest technical asset actually is: the shot data itself, if you get any.

The council will fixate on "our simulator is unvalidated" as a risk to manage down. I'd flip it: the simulator + benchmark harness is a reusable instrument, and the highest-upside move in the next two weeks is to stop treating "email CPRI for real data" as a background task and make it the centerpiece of the pitch, not a footnote in §8. Right now it's a drafted-but-unsent email. That's the single biggest thing left on the table. If CPRI hands over even 20-30 real shots, you can re-validate against ground truth instead of a simulator you built, and that's the difference between "promising decision-support tool" and "the tool that would've shipped with their next test campaign." Send it today, not in two weeks.

Second upside nobody's pricing in: this generalizes past 400kV insulator testing. The physics-calibration + memory architecture — exact circuit solve, deterministic IEC compliance, then a learned residual correction, then a cross-session memory that halves climbs — is a template for any impulse/pulse-testing decision problem: HVDC valve testing, surge arrester qualification, cable testing, even EMC pulse work. CPRI runs multiple halls beyond just this one generator. If you frame the demo as "here's a decision-support architecture we validated on your Track 1 generator, portable to your other test benches," you're not pitching a hackathon deliverable, you're pitching CPRI a reusable internal tool — which is exactly the kind of thing that turns a competition win into a post-hackathon relationship (internship offers, a pilot deployment, a real dataset partnership). Say this explicitly in the demo's closing slide: "this same shot-loop plus memory pattern applies to every impulse-type test CPRI runs, not just this generator." That reframes the judges from "grading a project" to "evaluating a vendor."

Third: the negative result (black-box BO loses to a technician) is underused as an asset. Most teams will show only wins. Showing a documented failure mode, framed as "this is why physics-first architectures beat black-box ML for safety-critical high-voltage decisions," makes you the only team that looks like it understands when NOT to use ML — a more sophisticated pitch to engineers than another accuracy table.

Bottom line for the two weeks: (1) send the CPRI email now, chase real data harder than any other workstream, (2) add one explicit "this generalizes beyond this generator" framing slide, (3) lean into the BO negative result as a selling point, not a hedge.

### The Outsider

Core argument: the biggest risk is a mismatch between what the team built and what the brief literally says they'll be judged on. CPRI's brief states judging is on "accuracy, precision, MAPE, MSE, RMSE, and which algorithm/tuning performs best... framed as a regression problem" — i.e., a bake-off against their kNN on their own data. The team's best result on that exact, literal task is the weakest number in their whole packet: a "narrow" 4-6% relative MAPE gain that doesn't survive Bonferroni correction, attributed to noisy residuals. Meanwhile their headline, most-quoted result (p<0.0001 shots-to-compliance, p=0.0007 climbs) comes from a simulator and a "technician" baseline they built themselves — which, to a naive reader, looks uncomfortably close to grading your own homework, even though the team is clearly aware of this and hedges carefully throughout. The concern: expert judges on a tight 24-hour clock will likely anchor on the literal rubric first, and the team's strongest narrative doesn't directly answer it.

Named the unsent CPRI email as the tell: it's been drafted but not sent for what sounds like most of the two weeks already spent, while the team kept building (overshoot predictor, browser memory system, atmospheric correction). Argued this is backwards — it's the only lever that could convert "narrow gain on noisy synthetic data" into something real, and there's no evidence it's been hard, just deprioritized versus building more.

Highest-leverage recommendation for the remaining ~2 weeks: (1) send the CPRI email today, not later, and (2) restructure the opening pitch to lead with the literal regression rubric result (their dataset, their kNN, their metrics) before the simulator/memory story, so a judge who only absorbs the first few minutes still sees the literal question answered.

### The Executor

**Odds:** Good, not great. This is the most technically serious entry in the room by a wide margin, and the CLAUDE.md shows a team that already killed the two things that usually sink hackathon ML projects — no honest baseline, and validating on your own model. But "best engineering" doesn't automatically win a room of domain-expert judges. What wins Monday morning is: can a CPRI engineer put in a real insulation spec and watch it work, live, in under two minutes, without you touching a keyboard.

**Biggest unseen risk: the offline app is the actual demo artifact, and it's the least-tested part of the whole system.** Every number in your evidence (p<0.0001, hybrid beats kNN, etc.) is generated by Python — benchmark.py, regression.py. But the thing judges will physically touch is impulse_app.html, a hand-built JS reimplementation. build_app.py already runs cross-checks (60 hybrid vectors, memory-vs-design agreement) and "aborts on mismatch" — that's the right instinct, but it means the app has broken silently before. A judge clicking through the shot loop and hitting an edge case (weird insulator input, a stage count edge, a browser without localStorage) live, on stage, is a worse failure than any p-value gap.

**Do Monday morning:** Stop adding models. Spend day 1 of the two weeks doing nothing but adversarial manual testing of impulse_app.html — every equipment type, every wrong input, cold browser cache, no localStorage, both impulse kinds, both light/dark. Then script the exact 5-minute live demo sequence from §9 word-for-word and rehearse it 10 times until it's boring. A boring flawless demo of a good result beats an exciting demo that has one crash.

## Anonymization mapping (revealed)

| Anonymous ID | Advisor |
|---|---|
| Response A | The Executor |
| Response B | The Contrarian |
| Response C | The Outsider |
| Response D | The First Principles Thinker |
| Response E | The Expansionist |

## Peer reviews (each advisor reviewed all five anonymized responses)

### Reviewer 1 (the Contrarian's seat)
**Strongest: C.** It's the only response that takes the stated judging criteria literally — MAPE/MSE/RMSE against CPRI's kNN, framed as regression — and points out the team's best answer to that exact question is their weakest number (4-6% relative MAPE, doesn't survive Bonferroni), while their headline, most-quoted result (p<0.0001) answers a different question nobody on the jury asked. The recommended fix is concrete and correctly sequenced for a 24-hour clock: send the CPRI email today, and restructure the opening pitch to lead with the literal rubric result before the simulator/memory story. D makes a genuinely good distinct catch (hardware ambiguity) but overstates it as the single biggest risk — CLAUDE.md already reasons through why the nameplate is the stronger evidence, so this is a real but bounded risk, not a project-zeroing one.

**Biggest blind spot: E.** Asked for the biggest unseen risk, it pivots entirely to upside framing and never names a real external risk — it explicitly reframes "simulator unvalidated" as a non-issue rather than engaging with why a room of CPRI hall engineers would treat that as disqualifying. It's the weakest answer to the actual question asked, reading more like pitch-polish advice than risk analysis.

**What all five missed:** None flag CLAUDE.md's own §7 "Known problems" list as a live-demo liability. Specifically, P2 — lab memory underperforms current practice at severity 1 (90% vs 100% first-shot), explicitly logged as "acceptance criterion met, symptom still visible, nothing fixed" — is exactly the kind of soft spot a CPRI engineer who runs impulse halls would notice and probe on stage, since the team's own documentation surfaces it. All five treated "biggest risk" as an external positioning or framing problem but none prepared the team for defending its own documented open questions (P1-P8) under direct technical Q&A from domain experts — treating risk as an artifact/narrative problem rather than a live cross-examination problem.

### Reviewer 2 (the First Principles Thinker's seat)
**Strongest: D.** It names the one risk that's concrete, checkable, and catastrophic if wrong — 12-stage/0.125µF nameplate vs 15-stage/3µF Excel calculator — and correctly ties it to every downstream number (resistors, climbs, the whole benchmark). C makes the same rubric-literalism point but stays abstract; D hands judges a single fact that can zero the project in five minutes, which is sharper than "lead with the literal metric."

**Biggest blind spot: E.** It reframes genuine risk (unvalidated simulator, unresolved hardware, self-graded headline result) as upside — "reusable architecture," "generalizes to other test benches," "lean into the BO loss as a selling point." That's narrative dressing over the same substance A/B/C/D treat as a problem, and it never engages with the hardware ambiguity or the rubric-literalism gap at all. In front of skeptical domain-expert judges, unearned reframing reads worse than an admitted weakness.

**What all five missed:** the "24-hour final" on 10-11 October is an on-site event with CPRI engineers physically present. The fastest way to resolve the nameplate-vs-Excel ambiguity (and open questions like units per stage, connection topology, which IEC edition/levels CPRI uses) isn't only "send the email today" — it's walking in with a short, specific question list to hand the judges themselves the moment the event opens, since an in-person answer lands faster than an inbox reply and de-risks the single biggest unknown before the pitch even starts. None of the five distinguish the pre-event prep window from the live 24-hour window, or note that judges being present is itself a resource the team hasn't planned to use.

### Reviewer 3 (the Expansionist's seat)
**Strongest: D.** It converts the rubric-mismatch point (also made by B and C) into the single most concrete, binary, catastrophic risk: not knowing which generator is real. Unlike C's more diffuse "lead with the literal number" fix, D's risk is falsifiable in one email reply and would zero out the entire physics stack if wrong — the sharpest, most checkable claim of the five, and it still lands the "send the email" recommendation everyone converges on.

**Biggest blind spot: E.** It reframes every risk the others raised (unvalidated simulator, rubric mismatch, hardware ambiguity) as upside framing instead of engaging them. "Lean into the BO loss as a selling point" and "this generalizes to other test benches" are positioning moves that assume the core technical risk is already handled — E never checks that assumption, and never even mentions the hardware ambiguity or rubric-literalism problem C and D center on. It reads as pitch coaching wearing risk-analysis clothes.

**What all five missed:** none has a contingency for CPRI not replying in time — they all say "send the email today" as if that resolves the risk, but with two weeks left and no guaranteed response window, the likely case is silence, and nobody says what to present if that happens. Also, none flag that the fitted circuit parameters (loop inductance, divider response) are explicitly non-identifiable per the project's own CLAUDE.md §7 — a HV engineer asking "what did it learn about my hall" is a live-demo landmine that none of the five named, despite it being a bigger technical vulnerability than most of what they did flag.

### Reviewer 4 (the Outsider's seat)
**Strongest: D.** It lands two concrete, falsifiable points instead of one mood: (a) the rubric-literal scoreboard is the 4-6% MAPE number, not the simulator story, and (b) the unconfirmed 12-stage/0.125µF vs 15-stage/3µF hardware question is a specific, checkable fact that can zero every resistor/climb number if wrong. C makes the first point too but stops there; D's hardware catch is sharper and matches CLAUDE.md's own P6 flag.

**Biggest blind spot: E.** It reframes real weaknesses (unvalidated sim, hardware ambiguity) as marketing opportunities without answering whether the core deliverable is even built on the right machine. It's the only response that doesn't treat the hardware question as urgent — it buries "send the email" under two spin-heavy bullet points instead of leading with it.

**All five missed:** CPRI's own 2000-row dataset is itself hardware-ambiguous — CLAUDE.md's P6 notes the dataset's SI resistor values (9.4-21.8 kΩ front) can't be built from either candidate machine's stock. That means the team's best literal-rubric result (beating CPRI's kNN on CPRI's own data) is standing on the same unresolved hardware question as everything else — nobody connected those two facts. Also missed: an email sent today likely won't get a reply before Oct 10-11, so "send the email" isn't a fix, it's a hedge — the team needs a plan for presenting under both hardware assumptions if no answer arrives, since judges may ask the nameplate question live as a test.

### Reviewer 5 (the Executor's seat)
**Strongest: A.** It's the only response that identifies a risk distinct from "send the CPRI email" — that impulse_app.html, not the Python benchmark, is what judges physically touch, and it's the least battle-tested artifact despite being the demo. That's a concrete, testable, two-week-actionable insight (adversarial UI testing, scripted rehearsal) that C/D/E don't raise at all.

**Biggest blind spot: E.** It reframes real risks (unvalidated simulator, rubric mismatch) as upside ("reusable architecture," "generalizes to HVDC/EMC") without engaging the actual danger — that on the literal judged task (accuracy vs CPRI's kNN, their data) the result is weak and Bonferroni-fails. E is pitch-polishing, not risk assessment; it under-weights that judges score the narrow regression bake-off, not the vision.

**All five missed:** the brief explicitly says judging includes "which algorithm and tuning performs best" — none asked whether hyperparameter tuning itself (grid search, tuning curves, ablations) was done and documented, versus just comparing default-setting models. Also unaddressed: no contingency plan if CPRI never answers the unsent email in two weeks — every response says "send it today" but none plans for silence.

## Chairman synthesis (verdict)

### Where the council agrees
- All five advisors independently named the same highest-leverage move: send the CPRI data-request email today. Zero disagreement, and it's been sitting drafted, not sent.
- The honest framing holds: current practice works in a well-characterised hall (100% first-shot) and falls apart in a poorly-characterised one (28–40%).
- The project's best answer to the literal judged task (MAPE/MSE/RMSE vs CPRI's own kNN, on CPRI's own data) is real but narrow (4–6% relative gain, fragile after Bonferroni correction), and must not be buried under the flashier, self-validated simulator story.
- The honest negative result (black-box BO loses to a technician) and the documented approximations are project strengths in front of domain-expert judges, not things to hide.
- Nobody thinks two more weeks should go into new models or features — every plan is consolidation, not addition.

### Where the council clashes
Three advisors (Contrarian, First Principles, Outsider) converge on the same risk from the same angle: the pitch's narrative leads with the wrong result. The Executor names an orthogonal risk: the offline demo app is the least-tested part of the system. The Expansionist doesn't engage with "risk" as asked at all — it reframes the same weaknesses as unclaimed upside. Four of five peer reviewers, independently, flagged the Expansionist's response as the weakest answer to the actual question — the closest thing to a consensus minority report the council produced.

### Blind spots the council caught (surfaced only in peer review)
- Sending the email isn't the same as getting an answer — nobody has a plan for silence, or a reply arriving after 10 October.
- The hardware ambiguity contaminates both stories, not just the flashy one: CPRI's own 2000-row dataset has switching-impulse resistor values that can't be built from either candidate generator's stock, so the narrow win and the big win both rest on the same unresolved assumption.
- The fitted circuit parameters (loop inductance, divider response) are a live cross-examination landmine — non-identifiable by the project's own admission, with no prepared answer for "what did it learn about my hall."
- The project's own documented weak spot (severity-1 first-shot dip to 90%, logged as "nothing fixed") could be found and used against the team by a judge who reads the documentation.
- Whether hyperparameter tuning/ablations were actually done and documented is untested, despite the rubric explicitly naming "tuning" as a judged axis.
- The judges are a resource on-site, not just an inbox — a short written question list handed to a judge in person on the event's opening morning resolves the hardware question faster than an email that may go unanswered.

### The recommendation
Restructure the pitch to lead with the literal, narrow, rubric-answering result first, stated plainly as real but modest. Bring the simulator and memory work in second, explicitly labeled as a validated architecture with no real-hall data yet. Send the CPRI email today, but prepare a short written question list (generator identity first) to hand judges in person at the event's opening. Spend real time adversarially testing impulse_app.html itself. Do not add another model or feature between now and the final.

### The one thing to do first
Today: send the CPRI data-request email — and in the same sitting, draft the short written question list (which generator is real, first) to hand judges in person on the morning the event opens. The email might go unanswered. The in-person window won't.
