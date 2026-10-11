"""
circuit.py — the impulse generator, as physics. No machine learning here.

A Marx generator of n stages, erected, is equivalent to a single-stage
circuit:

    C1 (charged to V0) --+--[ L ]--[ Rf ]--+-- C2 (test object + divider + strays)
                         |                  |
                        Rt                 (measurement point, via divider)
                         |                  |
                        gnd                gnd

    C1 = C_stage / n        V0 = n * V_charge
    Rf = n * Rf_stage       Rt = n * Rt_stage       L = n * L_stage + L_ext

State x = [v1, i, v2, vm]:
    C1 dv1/dt = -v1/Rt - i
    L  di/dt  =  v1 - Rf*i - v2
    C2 dv2/dt =  i
    tm dvm/dt =  v2 - vm          (divider response; tm = 0 means ideal)

The circuit is linear and time-invariant, so it is solved EXACTLY by
eigendecomposition rather than by a numerical ODE integrator. That makes a
single waveform a few milliseconds, which is what lets the design search
evaluate thousands of candidate settings.

Reference for the circuit and the approximations checked below:
Kuffel, Zaengl & Kuffel, High Voltage Engineering Fundamentals, ch. 2.
"""
from dataclasses import dataclass, replace
import numpy as np


# ---------------------------------------------------------------------------
# Hardware: what a 400 kV-class insulator test lab actually has
# ---------------------------------------------------------------------------
@dataclass
class Generator:
    """CPRI's impulse generator: HAEFELY 508085/E, 2400 kV, 12 stages.

    From the technical-data nameplate photographed on page 4 of CPRI's
    problem statement (HV IG Problem Statement.pdf):

        max charging voltage 2400 kV      stages 12       stage voltage 200 kV
        impulse capacitance 0.125 uF/stage, 2.5 kJ/stage, 30 kJ total
        front resistors per stage   30 ohm, 465 ohm, 3.7 kohm
        tail resistor per stage     520 ohm (lightning), 22 kohm (switching)
        charging resistor 135 kohm      basic load capacitance 545 pF
        test voltage range 50-2400 kV, minimum 2 active stages

    This replaces an earlier "3 MV, 15 stages, 0.375 uF" model built from
    briefing notes and a guessed capacitor value. The nameplate is the
    stronger evidence. CPRI's Excel calculator says 15 stages and 3 uF/stage,
    which contradicts the nameplate — flagged to CPRI, not modelled.

    The nameplate agrees with the physics: rt_stage = T2 / (0.7 * c_stage)
    predicts 571 ohm for a 50 us tail, and the plate says 520 ohm.

    Stock is expressed as PHYSICAL UNITS per stage. The values a design may
    use (rf_li, rt_li, rf_si, rt_si) are those units and every series or
    parallel pair of them — resistor_options(). With only three front values
    no single unit makes a 1.2 us front across the load range, so the
    combinations are required, not optional (CLAUDE.md P0c).

    UNCONFIRMED: how many units of each value a stage can take. Pairs assume
    at least two of each; CPRI's Excel inventory lists four per stage for a
    different stock. The per-stage and external loop inductance are the
    textbook values, not from the plate. Both are declared in CLAUDE.md.
    """
    n_stages: int = 12
    min_stages: int = 2               # nameplate: minimum 2 active stages
    c_stage: float = 0.125e-6         # F
    v_stage_max: float = 200e3        # V
    l_stage: float = 2.5e-6           # H, internal inductance per stage
    l_ext: float = 5.0e-6             # H, nominal external loop
    charge_margin: float = 0.90       # never charge above 90% of rating

    # Physical resistor units per stage, from the nameplate. Any unit may be
    # used in either impulse: physics decides (a 3.7 kohm front can never make
    # 1.2 us, a 30 ohm one never 250 us) rather than a rule we would invent.
    front_units: tuple = (30, 465, 3700)
    tail_li_units: tuple = (520,)
    tail_si_units: tuple = (22000,)
    cload_si: tuple = (0.0, 1e-9, 2e-9, 4e-9, 6e-9, 8e-9)   # added load, F

    # Fixed capacitance at the output
    c_base_load: float = 545e-12      # F, nameplate "basic load capacitance"
    c_divider: float = 500e-12        # F, CPRI's Excel default divider value
    c_stray_nominal: float = 150e-12  # F, CPRI's Excel default stray value

    def __post_init__(self):
        # What a design may actually set: the units and their pairs.
        opts = lambda units: tuple(sorted(resistor_options(units)))
        self.rf_li = self.rf_si = opts(self.front_units)
        self.rt_li = opts(self.tail_li_units)
        self.rt_si = opts(self.tail_si_units)

    @property
    def c_fixed(self):
        """Output capacitance that is neither test object nor stray."""
        return self.c_base_load + self.c_divider


def resistor_options(stock):
    """Every per-stage resistance reachable from the stock — a single
    stocked unit, or two of them combined in series or parallel.

    CPRI's briefing asks explicitly for "resistor combinations, including
    series-parallel combinations", and on the nameplate stock they are
    essential (see Generator). Pairs only: that needs two units of a value
    per stage, which is the least inventory the combinations can assume. The
    real count per stage is unconfirmed; CPRI's Excel lists four.

    Returns {resistance_ohm: "how to build it"}, deduplicated to 3
    significant figures so near-identical combinations collapse to one.
    """
    import itertools as _it

    def key(v):
        return round(v, 2 - int(np.floor(np.log10(v)))) if v > 0 else 0.0

    # A pair that moves its dominant unit by less than a unit's own tolerance
    # (lab.py draws each resistor within +-5%) is the same resistor in
    # practice — nobody bolts 3.7 kohm across 30 ohm for a 0.8% change. Such
    # pairs are dropped rather than offered as a "different" setting.
    TOL = 0.05
    out = {}
    for r in stock:
        out.setdefault(key(r), f"single {r:g} ohm unit")
    for r1, r2 in _it.combinations_with_replacement(stock, 2):
        s, p = r1 + r2, r1 * r2 / (r1 + r2)
        if s / max(r1, r2) - 1 >= TOL:
            out.setdefault(key(s), f"{r1:g} ohm + {r2:g} ohm in series")
        if 1 - p / min(r1, r2) >= TOL:
            out.setdefault(key(p), f"{r1:g} ohm || {r2:g} ohm in parallel")
    return out


@dataclass
class Setting:
    """What an operator actually chooses."""
    n: int                 # stages in series, per column
    v_charge: float        # V per stage
    rf_stage: float        # ohm
    rt_stage: float        # ohm
    c_load: float = 0.0    # F, added load capacitance (switching impulse)
    p: int = 1             # parallel columns; p=1 is the plain all-in-series
                           # erection every benchmarked result uses


@dataclass
class Params:
    """Everything the circuit needs, in SI units."""
    c1: float
    c2: float
    rf: float
    rt: float
    l: float
    v0: float
    tm: float = 0.0


def build_params(s: Setting, g: Generator, c_test: float, *, l_factor=1.0,
                 l_extra=0.0, c_stray=None, tm=0.0, r_tol=None,
                 c_stage_tol=1.0) -> Params:
    """Assemble circuit parameters from a setting and the hardware.

    The keyword arguments are how the LAB differs from the textbook: extra
    lead inductance, a different stray capacitance, resistor tolerance, a
    non-ideal divider. The design model uses defaults; the lab simulator in
    lab.py supplies its own hidden values.
    """
    c_stray = g.c_stray_nominal if c_stray is None else c_stray
    rf_mult, rt_mult = (1.0, 1.0) if r_tol is None else r_tol
    # p columns of n series stages: capacitance and resistance scale with the
    # series count n and divide / multiply by the parallel count p. With p = 1
    # every expression below is the one the benchmark was run with.
    return Params(
        c1=g.c_stage * c_stage_tol * s.p / s.n,
        c2=c_test + g.c_fixed + c_stray + s.c_load,
        rf=s.n * s.rf_stage * rf_mult / s.p,
        rt=s.n * s.rt_stage * rt_mult / s.p,
        l=(s.n * g.l_stage / s.p + g.l_ext) * l_factor + l_extra,
        v0=s.n * s.v_charge,
        tm=tm,
    )


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------
def simulate(p: Params, t: np.ndarray) -> np.ndarray:
    """Voltage at the measurement point, exactly, at times t (seconds)."""
    l = max(p.l, 1e-9)
    A = np.array([
        [-1 / (p.rt * p.c1), -1 / p.c1, 0.0],
        [1 / l, -p.rf / l, -1 / l],
        [0.0, 1 / p.c2, 0.0],
    ])
    x0 = np.array([p.v0, 0.0, 0.0])
    if p.tm > 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        A[3, 2], A[3, 3] = 1 / p.tm, -1 / p.tm
        x0 = np.r_[x0, 0.0]

    lam, V = np.linalg.eig(A)
    coef = np.linalg.solve(V, x0.astype(complex))
    out_row = V[-1, :] if p.tm > 0 else V[2, :]
    # v(t) = sum_k V[row,k] * coef_k * exp(lam_k t)
    wave = (np.exp(np.outer(t, lam)) * (out_row * coef)).sum(axis=1)
    return wave.real


def time_axis(kind: str):
    """Sampling that resolves the physics of each impulse type."""
    if kind == "LI":
        return np.arange(0, 120e-6, 10e-9)      # 10 ns over 120 us
    return np.arange(0, 7000e-6, 2e-6)           # 2 us over 7 ms


# ---------------------------------------------------------------------------
# Textbook approximations (Kuffel ch. 2) — used as self-checks and as a
# first guess, never as the answer.
# ---------------------------------------------------------------------------
def approx_front_time(p: Params):
    return 3.0 * p.rf * p.c1 * p.c2 / (p.c1 + p.c2)


def approx_tail_time(p: Params):
    """Kuffel gives two forms. With the discharge resistor across the LOAD
    it is 0.7 (Rf + Rt)(C1 + C2). With it across the GENERATOR capacitance —
    the topology modelled here, and the usual one because it gives better
    efficiency — it is 0.7 Rt (C1 + C2). Using the wrong one is a 27% error
    on this generator."""
    return 0.7 * p.rt * (p.c1 + p.c2)


def approx_efficiency(p: Params):
    return p.c1 / (p.c1 + p.c2)


def critical_front_resistance(p: Params):
    """Below this, the stray inductance rings. The practical conflict in
    lightning impulse testing: a 1.2 us front wants Rf near this value, so
    overshoot and front time pull against each other."""
    ce = p.c1 * p.c2 / (p.c1 + p.c2)
    return 2.0 * np.sqrt(p.l / ce)


# ---------------------------------------------------------------------------
def _checks():
    from waveform import evaluate
    ok = True

    def chk(name, got, lo, hi, unit=""):
        nonlocal ok
        p = lo <= got <= hi
        ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}] {name:58s} {got:9.4f}{unit} "
              f"(expect {lo}-{hi})")

    g = Generator()
    t = time_axis("LI")
    # Near-zero inductance so the textbook approximations should hold
    s = Setting(n=12, v_charge=100e3, rf_stage=30, rt_stage=520)
    p = build_params(s, g, c_test=100e-12, l_factor=1e-4)
    r = evaluate(t, simulate(p, t), "LI", base_curve=simulate(p, t))

    print("Circuit vs Kuffel approximations (negligible inductance)")
    chk("front time: exact / 3*Rf*Ce", r.t1 / (approx_front_time(p) * 1e6),
        0.90, 1.10)
    chk("tail time: exact / 0.7*Rt*(C1+C2)  (Rt across C1)",
        r.t2 / (approx_tail_time(p) * 1e6), 0.90, 1.10)
    chk("efficiency: exact / C1/(C1+C2)",
        (r.extreme / p.v0) / approx_efficiency(p), 0.90, 1.02)

    print("\nPhysics")
    p2 = replace(p, v0=2 * p.v0)
    chk("linear in charging voltage (2x V0 -> 2x peak)",
        simulate(p2, t).max() / simulate(p, t).max(), 1.999, 2.001)

    pl = build_params(s, g, c_test=100e-12)          # realistic inductance
    rc = critical_front_resistance(pl)
    under = replace(pl, rf=0.4 * rc)
    over = replace(pl, rf=1.5 * rc)
    base_u = simulate(replace(under, l=1e-9), t)
    base_o = simulate(replace(over, l=1e-9), t)
    ru = evaluate(t, simulate(under, t), "LI", base_curve=base_u)
    ro = evaluate(t, simulate(over, t), "LI", base_curve=base_o)
    chk("underdamped (Rf = 0.4 Rcrit) rings: overshoot", ru.overshoot,
        0.05, 0.60)
    chk("overdamped (Rf = 1.5 Rcrit) does not", ro.overshoot, 0.0, 0.01)
    # Rcrit = 2*sqrt(L/Ce): 35 uH of loop at 12 stages into ~1.15 nF of Ce
    # (C1 = 10.4 nF in series with ~1.3 nF of output) gives ~349 ohm total.
    # The band is machine-specific; re-derive it if the generator changes.
    chk("critical front resistance, this setup", rc, 300, 400, " ohm")

    print("\nDivider response")
    pm = replace(pl, tm=50e-9)
    rm = evaluate(t, simulate(pm, t), "LI",
                  base_curve=simulate(replace(pm, l=1e-9), t))
    rn = evaluate(t, simulate(pl, t), "LI",
                  base_curve=simulate(replace(pl, l=1e-9), t))
    chk("slow divider lengthens the measured front", rm.t1 / rn.t1,
        1.001, 1.30)

    print("\nResistor combinations (P0c - series/parallel from declared stock)")
    opts = resistor_options((100.0, 100.0, 220.0))
    chk("single units pass through unchanged", 100.0 in opts, 1, 1)
    chk("series of two equal units (100+100=200)", 200.0 in opts, 1, 1)
    chk("parallel of two equal units (100||100=50)", 50.0 in opts, 1, 1)
    chk("series of unequal units (100+220=320)", 320.0 in opts, 1, 1)
    chk("wider than the raw stock alone", len(opts) > 3, 1, 1)
    far = resistor_options((30.0, 3700.0))
    chk("pair within unit tolerance dropped (30||3700 = 29.8)",
        29.8 not in far and 3730.0 not in far, 1, 1)

    print("\nSeries-parallel stage connection")
    ps = build_params(Setting(n=6, v_charge=100e3, rf_stage=30, rt_stage=520), g, 100e-12)
    pp = build_params(Setting(n=6, v_charge=100e3, rf_stage=30, rt_stage=520, p=2), g, 100e-12)
    chk("2 parallel columns double the generator capacitance", pp.c1 / ps.c1, 1.999, 2.001)
    chk("... and halve the effective front resistance", pp.rf / ps.rf, 0.499, 0.501)
    chk("... and leave the output voltage at n x V", pp.v0 / ps.v0, 0.999, 1.001)
    p1 = build_params(Setting(n=12, v_charge=100e3, rf_stage=30, rt_stage=520, p=1), g, 100e-12)
    p1b = build_params(Setting(n=12, v_charge=100e3, rf_stage=30, rt_stage=520), g, 100e-12)
    chk("p = 1 is exactly the all-in-series circuit", float(p1 == p1b), 1, 1)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _checks() else 1)
