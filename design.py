"""
design.py — given a target waveform and what we believe about the lab,
recommend generator settings from the hardware actually in stock.

The circuit is linear, so waveform SHAPE does not depend on charging
voltage. We therefore search shape over (stages, front resistor, tail
resistor, load capacitor) at unit charge, then set the charging voltage
analytically to hit the target test voltage. That turns a 4-D continuous
problem into an exhaustive search over a few thousand discrete candidates,
each evaluated exactly.

Among compliant candidates we prefer:
  1. the largest distance inside every tolerance band (robust to model
     error — a design sitting on the edge fails on the real generator), then
  2. charging voltage in the comfortable 50-90% of rating band, because
     spark gaps fire unreliably at low charge and over-stress at high charge.
"""
from dataclasses import dataclass, replace
import itertools

from circuit import (Generator, Setting, Params, build_params, simulate,
                     time_axis)
from waveform import evaluate


@dataclass
class Belief:
    """What the designer thinks the lab is. The textbook belief is the
    nominal generator; calibration and lab memory refine it."""
    l_factor: float = 1.0
    l_extra: float = 0.0
    c2_offset: float = 0.0        # F, added to the modelled load capacitance
    c_test_scale: float = 1.0     # multiplies the test object estimate
    tm: float = 0.0
    eff_scale: float = 1.0        # measured / modelled efficiency


def model_params(s: Setting, g: Generator, c_test_nominal: float,
                 b: Belief) -> Params:
    p = build_params(s, g, c_test_nominal * b.c_test_scale,
                     l_factor=b.l_factor, l_extra=b.l_extra, tm=b.tm)
    return replace(p, c2=p.c2 + b.c2_offset)


def predict_fast(s, g, c_test_nominal, b, kind):
    """Screening only: zero-inductance base curve, no fitting. Can misjudge
    front time by ~10% at high loop inductance, so it is never used to make
    the final choice — only to shortlist."""
    t = time_axis(kind)
    p = model_params(s, g, c_test_nominal, b)
    u = simulate(p, t)
    base = simulate(replace(p, l=1e-9), t) if kind == "LI" else None
    return evaluate(t, u, kind, base_curve=base), t, u


def predict(s: Setting, g: Generator, c_test_nominal: float, b: Belief,
            kind: str):
    """Predict what the LAB'S EVALUATOR will report for this shot.

    An earlier version used the circuit's zero-inductance response as the
    base curve, because it is the physically "exact" mean curve. That is not
    how IEC 60060-1 defines the base curve — it is a fitted double
    exponential — and with ~120 uH of loop inductance the two differ by ~12%
    in front time. The design then rejected the only compliant settings in
    stock. The prediction must replicate the standard's procedure, not the
    physics, because compliance is judged by the procedure.
    """
    t = time_axis(kind)
    p = model_params(s, g, c_test_nominal, b)
    u = simulate(p, t)
    r = evaluate(t, u, kind)            # same IEC procedure as the lab
    return r, t, u


def candidates(kind: str, g: Generator, p: int = 1):
    """Every setting the hardware allows. The resistor values already include
    series/parallel pairs of the stocked units (Generator.__post_init__), and
    stage count runs from the nameplate minimum of 2. With p > 1 the whole
    bank is wired as p parallel columns of n_stages / p series stages (the
    nameplate's "p s" connection); 12 stages give 2p6s, 3p4s, 4p3s, 6p2s."""
    if p > 1:
        if g.n_stages % p or g.n_stages // p < g.min_stages:
            return iter(())
        stages = [g.n_stages // p]
    else:
        stages = range(g.min_stages, g.n_stages + 1)
    if kind == "LI":
        return itertools.product(stages, g.rf_li, g.rt_li, (0.0,))
    return itertools.product(stages, g.rf_si, g.rt_si, g.cload_si)


@dataclass
class Correction:
    """Measured-vs-predicted residual at shots already fired, applied to
    every candidate's prediction. The physics says how the waveform moves
    between settings; the residual says where this hall actually sits. It
    absorbs what the model cannot represent — resistor tolerance, divider
    detail, evaluation noise."""
    t1: float = 1.0
    t2: float = 1.0
    overshoot: float = 0.0


def design(kind, target_v, c_test_nominal, b: Belief, g: Generator = None,
           top=5, restrict=None, corr: Correction = None, parallel=(1,)):
    """Ranked list of settings. `restrict` limits the search, e.g. to one
    stage count when the operator does not want to re-erect the generator."""
    g = g or Generator()
    pool = [c + (1,) for c in candidates(kind, g)
            if not restrict or restrict(*c)]
    for p_cols in parallel:               # (1,) = the benchmarked search
        if p_cols > 1:
            pool += [c + (p_cols,) for c in candidates(kind, g, p_cols)
                     if not restrict or restrict(*c)]
    # Two stages: screen everything fast, then evaluate a generous shortlist
    # with the exact IEC procedure. Small searches skip the screen.
    if kind == "LI" and len(pool) > 80:
        scored = []
        for n, rf, rt, cl, pc in pool:
            s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl, p=pc)
            r, _, _ = predict_fast(s, g, c_test_nominal, b, kind)
            scored.append((r.margin(target_v, shape_only=True), (n, rf, rt, cl, pc)))
        scored.sort(key=lambda x: x[0])
        pool = [c for _, c in scored[:60]]

    rows = []
    for n, rf, rt, cl, pc in pool:
        s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl, p=pc)
        r, _, _ = predict(s, g, c_test_nominal, b, kind)
        per_volt = r.test_voltage * b.eff_scale
        if per_volt <= 0:
            continue
        v_ch = target_v / per_volt
        frac = v_ch / g.v_stage_max
        if frac > g.charge_margin:
            continue
        if corr is not None:
            r.t1 *= corr.t1
            r.t2 *= corr.t2
            r.overshoot = max(0.0, r.overshoot + corr.overshoot)
        shape = r.margin(target_v, shape_only=True)
        charge_pen = 0.0 if 0.5 <= frac <= 0.9 else abs(frac - 0.7)
        rows.append({
            "setting": replace(s, v_charge=float(v_ch)),
            "shape_margin": shape, "charge_frac": float(frac),
            "t1": r.t1, "t2": r.t2, "overshoot": r.overshoot,
            "score": shape + 0.5 * charge_pen,
        })
    rows.sort(key=lambda d: d["score"])
    return rows[:top]


if __name__ == "__main__":
    import time
    g = Generator()
    for kind, v in [("LI", 1425e3), ("SI", 1050e3)]:
        t0 = time.time()
        best = design(kind, v, 90e-12, Belief(), g)
        dt = time.time() - t0
        print(f"\n{kind} {v/1e3:.0f} kV  — searched in {dt:.1f} s")
        for d in best[:3]:
            s = d["setting"]
            print(f"  n={s.n:2d}  Rf={s.rf_stage:6.0f}  Rt={s.rt_stage:6.0f}  "
                  f"Cl={s.c_load*1e9:3.0f}nF  Vch={s.v_charge/1e3:6.1f}kV "
                  f"({d['charge_frac']*100:3.0f}%)  T1={d['t1']:8.2f}  "
                  f"T2={d['t2']:8.1f}  os={d['overshoot']*100:4.1f}%  "
                  f"margin={d['shape_margin']:.2f}")
