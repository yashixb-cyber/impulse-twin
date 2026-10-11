"""
lab.py — a stand-in for a real impulse hall. The design model never sees
inside this file.

Why it exists: if you design settings with a circuit model and then check
them against the SAME circuit model, you measure the model's agreement with
itself. Every result would look perfect and mean nothing. So "the lab" is a
higher-fidelity simulator with effects the design model does not include:

    - lead and loop inductance different from nominal (lab-specific, fixed)
    - stray capacitance to ground different from the assumed value
    - a non-ideal divider with its own response time
    - resistor tolerances, fixed per physical resistor in the lab's stock
    - stage capacitor tolerance
    - test-object capacitance that differs from its estimate, with a
      systematic bias per insulator type
    - measurement noise

The first four are properties of the LAB and stay constant across test
sessions — which is what lets cross-session learning help. The test-object
error changes every session.

This is still a simulator. Results measure robustness to this specified
class of mismatch, not performance on a real generator. That distinction is
stated wherever results are reported.
"""
from dataclasses import dataclass, field
import numpy as np

from circuit import Generator, Setting, build_params, simulate, time_axis

CATEGORIES = {
    # name: (nominal C range pF, typical estimate bias)
    "porcelain_disc_string": ((60, 120), +0.22),
    "composite_long_rod": ((30, 60), +0.06),
    "post_insulator_stack": ((120, 250), -0.10),
}


@dataclass
class TestObject:
    category: str
    c_nominal: float          # F, what the lab estimates
    kind: str                 # "LI" or "SI"
    target_v: float           # V


@dataclass
class Lab:
    """One impulse hall. All hidden."""
    seed: int = 0
    g: Generator = field(default_factory=Generator)

    def __post_init__(self):
        r = np.random.default_rng(self.seed)
        self.l_factor = float(r.uniform(0.85, 1.60))
        self.l_extra = float(r.uniform(4e-6, 14e-6))
        self.c_stray = float(r.uniform(100e-12, 280e-12))
        self.tm = float(r.uniform(15e-9, 60e-9))
        self.c_stage_tol = float(r.uniform(0.97, 1.03))
        self.noise = 0.003
        self.cat_bias = {k: float(v[1] + r.normal(0, 0.04))
                         for k, v in CATEGORIES.items()}
        # Each physical resistor in stock has its own fixed tolerance
        vals = set(self.g.rf_li + self.g.rt_li + self.g.rf_si + self.g.rt_si)
        self.r_tol = {v: float(r.uniform(0.95, 1.05)) for v in vals}
        self._rng = np.random.default_rng(self.seed + 10_000)

    def reseed_noise(self, *key):
        """Give the shots that follow their own measurement-noise stream.
        The benchmark reseeds per (session, strategy): with one shared
        stream, a change to how many shots one strategy fires shifted the
        noise every other strategy saw, and moved their results too."""
        self._rng = np.random.default_rng([int(k) for k in key])

    def true_c_test(self, obj: TestObject, session_seed: int):
        r = np.random.default_rng(session_seed)
        return obj.c_nominal * (1 + self.cat_bias[obj.category]) * \
            (1 + r.normal(0, 0.06))

    def fire(self, s: Setting, obj: TestObject, session_seed: int):
        """One shot. Returns the measured record (t in s, u in V)."""
        t = time_axis(obj.kind)
        p = build_params(
            s, self.g, self.true_c_test(obj, session_seed),
            l_factor=self.l_factor, l_extra=self.l_extra,
            c_stray=self.c_stray, tm=self.tm,
            r_tol=(self.r_tol.get(s.rf_stage, 1.0),
                   self.r_tol.get(s.rt_stage, 1.0)),
            c_stage_tol=self.c_stage_tol)
        u = simulate(p, t)
        u = u + self._rng.normal(0, self.noise * u.max(), len(u))
        return t, u


def make_session(rng, kind=None):
    """A realistic test request for 400 kV-class insulators."""
    cat = rng.choice(list(CATEGORIES))
    lo, hi = CATEGORIES[cat][0]
    kind = kind or rng.choice(["LI", "SI"], p=[0.65, 0.35])
    # IEC 60071 withstand levels for Um = 420 kV class — confirm the exact
    # values against the edition CPRI uses before quoting them.
    target = rng.choice([1300e3, 1425e3, 1550e3]) if kind == "LI" \
        else rng.choice([950e3, 1050e3])
    return TestObject(cat, float(rng.uniform(lo, hi)) * 1e-12, kind,
                      float(target))


def hall(seed, severity=1.0):
    """A lab whose departure from the nominal model scales with severity.

    severity 1 is a well-characterised hall; 4 is a large, poorly
    characterised setup — long leads, unknown strays, a slow divider, and
    insulator capacitance estimates that are well off. The severity sweep
    is how we find where the textbook design stops being enough.
    """
    lab = Lab(seed=seed)
    k = float(severity)
    lab.l_factor = 1.0 + (lab.l_factor - 1.0) * k
    lab.l_extra *= k
    lab.c_stray *= 1 + 0.6 * (k - 1)
    lab.tm *= k
    lab.cat_bias = {c: v * k for c, v in lab.cat_bias.items()}
    lab.severity = k
    return lab


def scaled_session(rng, severity=1.0, kind=None):
    obj = make_session(rng, kind)
    obj.c_nominal *= 1 + 1.5 * (float(severity) - 1)   # larger setups
    return obj
