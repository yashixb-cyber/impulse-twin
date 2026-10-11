"""
benchmark.py — which strategy reaches a compliant waveform in fewest shots,
and how does that change as the hall departs further from the textbook?

For every test session we also run an ORACLE that searches the hall's real
behaviour for any stocked setting that complies. Sessions where none exists
are reported separately — no strategy can be blamed for a target the
hardware cannot meet, and a benchmark that counted them would be measuring
the resistor stock, not the method.

Lab memory learns ONLINE: each session is predicted before its own shots
are added to memory, so every memory result is out-of-sample.

    python benchmark.py            # full sweep, writes benchmark_results.csv
    python benchmark.py --quick    # smaller sweep for a fast check
"""
import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from circuit import Generator, Setting
from lab import hall, scaled_session
from waveform import evaluate
import strategies as S

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

warnings.filterwarnings("ignore")


def oracle_feasible(lab, obj, seed, g):
    """Does ANY stocked setting comply in this hall, noise-free, with ideal
    voltage correction? Ground truth the strategies never see."""
    noise, lab.noise = lab.noise, 0.0
    rf_s, rt_s = (g.rf_li, g.rt_li) if obj.kind == "LI" else (g.rf_si, g.rt_si)
    cl_s = (0.0,) if obj.kind == "LI" else g.cload_si
    found = 0
    try:
        for n, rf, rt, cl in itertools.product(range(g.min_stages, g.n_stages + 1),
                                               rf_s, rt_s, cl_s):
            s = Setting(n, 100e3, rf, rt, cl)
            t, u = lab.fire(s, obj, seed)
            r = evaluate(t, u, obj.kind)
            vch = s.v_charge * obj.target_v / max(r.test_voltage, 1)
            if vch > g.charge_margin * g.v_stage_max:
                continue
            if r.margin(obj.target_v, shape_only=True) <= 1.0:
                found += 1
    finally:
        lab.noise = noise
    return found


def resistor_changes(shots):
    """How many times the operator actually had to climb the generator and
    change Rf or Rt between shots — Dr. Sheshu's stated pain point (P0e),
    not just the shot count. Pure post-processing over what each strategy
    already returns; touches no strategy code."""
    changes = 0
    for prev, cur in zip(shots, shots[1:]):
        p, c = prev["setting"], cur["setting"]
        if p.rf_stage != c.rf_stage or p.rt_stage != c.rt_stage:
            changes += 1
    return changes


def run(severities, halls_per, warmup, evals, kinds=("LI", "SI"), seed=0,
        halls=None, on_session=None):
    """`on_session(row, obj, shots_by_strategy)` sees every scored session's
    shots — dashboard.py replays one benchmark session through it."""
    g = Generator()
    rows = []
    t0 = time.time()
    for sev in severities:
        for h in (halls if halls is not None else range(halls_per)):
            hall_seed = 1000 * sev + h
            lab = hall(hall_seed, sev)
            rng = np.random.default_rng(hall_seed)
            memory = S.LabMemory()
            app_mem = S.AppMemory()

            # Warm-up: the hall's history before the sessions we score.
            for w in range(warmup):
                obj = scaled_session(rng, sev, kind=rng.choice(kinds))
                lab.reseed_noise(hall_seed * 100 + w, 99)
                sh = S.calibrated(lab, obj, seed=hall_seed * 100 + w)
                memory.add(obj, sh[-1]["setting"], sh[-1]["fits"][-1], g)
                app_mem.add(obj, sh, g)   # same history, four readings a shot

            for e in range(evals):
                obj = scaled_session(rng, sev, kind=rng.choice(kinds))
                sess = hall_seed * 100 + 500 + e
                feas = oracle_feasible(lab, obj, sess, g)
                row = {"severity": sev, "hall": hall_seed, "session": e,
                       "kind": obj.kind, "category": obj.category,
                       "feasible_settings": feas}
                kept = {}
                # Each strategy gets its own noise stream for this session, so
                # a change to one strategy cannot shift another's results.
                # New strategies go at the END to keep the others' keys.
                for k, (name, fn) in enumerate([
                        ("technician", lambda: S.technician(lab, obj, sess, g)),
                        ("blackbox_bo", lambda: S.blackbox_bo(lab, obj, sess, g)),
                        ("calibration", lambda: S.calibrated(lab, obj, sess, g)),
                        ("lab_memory", lambda: S.with_memory(lab, obj, sess,
                                                             memory, g)),
                        ("app_loop", lambda: S.app_loop(lab, obj, sess, g)),
                        ("app_memory", lambda: S.app_memory(lab, obj, sess,
                                                            app_mem, g))]):
                    lab.reseed_noise(sess, k)
                    sh = fn()
                    kept[name] = sh
                    row[f"{name}_shots"] = len(sh)
                    row[f"{name}_solved"] = bool(sh[-1]["compliant"])
                    row[f"{name}_first"] = bool(sh[0]["compliant"])
                    row[f"{name}_rchanges"] = resistor_changes(sh)
                    if name == "lab_memory":
                        memory.add(obj, sh[-1]["setting"], sh[-1]["fits"][-1], g)
                    if name == "app_memory":
                        app_mem.add(obj, sh, g)
                rows.append(row)
                if on_session:
                    on_session(row, obj, kept)
            print(f"  severity {sev} hall {h} done "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


STRATS = ["technician", "blackbox_bo", "calibration", "lab_memory", "app_loop",
          "app_memory"]


def summarise(df):
    feas = df[df.feasible_settings > 0]
    out = []
    for sev, grp in feas.groupby("severity"):
        rec = {"severity": sev, "sessions": len(grp)}
        for s in STRATS:
            solved = grp[f"{s}_solved"]
            rec[f"{s}_first_%"] = round(100 * grp[f"{s}_first"].mean(), 0)
            # Mean shots over ALL sessions, with an unsolved session counted
            # at the full budget. Averaging only over solved sessions rewards
            # a strategy for failing on the hard ones: black-box BO scored a
            # perfect-looking 1.00 that way while solving half the sessions.
            eff = np.where(solved, grp[f"{s}_shots"], S.MAX_SHOTS)
            rec[f"{s}_shots"] = round(float(eff.mean()), 2)
            rec[f"{s}_solved_%"] = round(100 * solved.mean(), 0)
            rec[f"{s}_rchanges"] = round(float(grp[f"{s}_rchanges"].mean()), 2)
        out.append(rec)
    return pd.DataFrame(out), int((df.feasible_settings == 0).sum())


def summarise_by_kind(df):
    """P4: the same summary, lightning and switching reported separately.
    The kinds are drawn at random per session (65% LI), so each half has
    fewer sessions than the pooled table — read it as a breakdown, not as
    a second set of results."""
    out = []
    for kind, grp in df.groupby("kind"):
        summ, _ = summarise(grp)
        out.append(summ.assign(kind=kind))
    cols = ["kind"] + [c for c in out[0].columns if c != "kind"]
    return pd.concat(out, ignore_index=True)[cols]


def paired_by_kind(df):
    """P4: the headline tests (vs current practice, severities 3-4, shots
    and climbs) run separately on lightning and switching sessions.
    Exploratory: NOT part of all_paired_tests' Bonferroni family, and each
    half has about a third or two thirds of the sessions."""
    rows = []
    for kind, grp in df.groupby("kind"):
        for metric in ("shots", "rchanges"):
            rows.append(paired_tests(grp, metric=metric).assign(kind=kind))
    out = pd.concat(rows, ignore_index=True)
    return out[["kind"] + [c for c in out.columns if c != "kind"]]


def all_paired_tests(df):
    """The headline test first, then the checks CLAUDE.md says to re-run
    whenever the generator or stock changes: resistor changes (the stated
    pain point) and P2, whether lab memory is ever worse than calibration
    alone, on easy halls and hard ones."""
    return pd.concat([
        paired_tests(df),
        paired_tests(df, metric="rchanges"),
        paired_tests(df, severities=(1, 2)),
        paired_tests(df, severities=(1, 2), base="calibration"),
        paired_tests(df, severities=(3, 4), base="calibration"),
        # does the app's memory add anything over the app loop without it?
        paired_tests(df, base="app_loop", only=("app_memory",)),
    ], ignore_index=True)


def _paired_metric(x, s, metric):
    if metric == "shots":      # unsolved counted at the budget
        return np.where(x[f"{s}_solved"], x[f"{s}_shots"], S.MAX_SHOTS)
    return x[f"{s}_{metric}"].to_numpy()


def paired_tests(df, severities=(3, 4), metric="shots", base="technician",
                 only=None):
    """Is each strategy genuinely better than `base` (current practice)?

    Every strategy runs on the same sessions, so the comparison is paired:
    for each session, did it need fewer shots than the technician, more, or
    the same? Wilcoxon signed-rank on the shot counts, unsolved sessions
    counted at the full budget. Without this, a 45% vs 60% first-shot figure
    on 20 sessions looks like a result and is not one. `metric="rchanges"`
    runs the same test on resistor changes; `fewer`/`more` count sessions.
    """
    from scipy.stats import wilcoxon
    x = df[(df.feasible_settings > 0) & df.severity.isin(severities)]
    rows = []
    ref = _paired_metric(x, base, metric)
    for s in STRATS:
        if s == base or (only and s not in only):
            continue
        cand = _paired_metric(x, s, metric)
        d = ref - cand
        p = float(wilcoxon(ref, cand).pvalue) if (d != 0).any() else 1.0
        rows.append({"strategy": s, "vs": base, "metric": metric,
                     "severities": "-".join(map(str, severities)),
                     "sessions": len(x),
                     "fewer": int((d > 0).sum()),
                     "more": int((d < 0).sum()),
                     "same": int((d == 0).sum()),
                     "wilcoxon_p": round(p, 4),
                     "verdict": ("better" if p < .05 and (d > 0).sum() > (d < 0).sum()
                                 else "worse" if p < .05 else "no significant difference")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--medium", action="store_true")
    a = ap.parse_args()
    if a.medium:
        df = run([1, 2, 3, 4], halls_per=2, warmup=10, evals=8)
    elif a.quick:
        df = run([1, 4], halls_per=1, warmup=8, evals=6)
    else:
        df = run([1, 2, 3, 4], halls_per=3, warmup=10, evals=10)
    df.to_csv(os.path.join(RESULTS, "benchmark_results.csv"), index=False)
    summ, infeasible = summarise(df)
    summ.to_csv(os.path.join(RESULTS, "benchmark_summary.csv"), index=False)

    print("\n" + "=" * 92)
    print("FIRST-SHOT IEC COMPLIANCE (%) — feasible sessions only")
    print("=" * 92)
    print(summ[["severity", "sessions"] +
               [f"{s}_first_%" for s in STRATS]].to_string(index=False))
    print("\nMEAN SHOTS TO COMPLIANCE (solved sessions)")
    print(summ[["severity"] + [f"{s}_shots" for s in STRATS]].to_string(index=False))
    print("\nSOLVED WITHIN 10 SHOTS (%)")
    print(summ[["severity"] + [f"{s}_solved_%" for s in STRATS]].to_string(index=False))
    pt = all_paired_tests(df)
    pt.to_csv(os.path.join(RESULTS, "benchmark_paired.csv"), index=False)
    print("\nPAIRED TESTS — first block is the headline: vs current "
          "practice, severities 3-4, shots")
    print(pt.to_string(index=False))
    print(f"\nsessions where NO stocked setting complies (excluded): {infeasible} "
          f"of {len(df)}")
    json.dump({"infeasible": infeasible, "total": len(df)},
              open(os.path.join(RESULTS, "benchmark_meta.json"), "w", encoding="utf-8"))
