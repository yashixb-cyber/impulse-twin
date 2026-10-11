"""
waveform.py — evaluate a recorded impulse the way IEC 60060-1 does.

This is deliberately NOT machine learning. The standard defines these
parameters deterministically, and a lab's result is only valid if it was
evaluated that way. Replacing any of this with a learned model would make
our answers non-standard.

Lightning impulse (IEC 60060-1:2010):
  1. Fit a double-exponential BASE CURVE to the record.
  2. Residual = record - base curve.
  3. Filter the residual with the TEST VOLTAGE FUNCTION
         k(f) = 1 / (1 + 2.2 f^2),   f in MHz
     applied as a zero-phase filter.
  4. TEST VOLTAGE CURVE = base curve + filtered residual.
  5. Test voltage, front time T1 and time to half T2 come from the test
     voltage curve.
         T1 = 1.67 (t90 - t30)          virtual origin O1 = t30 - 0.3 T1
         T2 = t50(tail) - O1
  6. Relative overshoot  beta = (Ue - Ub) / Ue
         Ue = extreme of the record, Ub = extreme of the base curve.

Switching impulse (IEC 60060-1:2010):
         Tp = K * T_AB,  T_AB = t90 - t30 on the front,
         K  = 2.42 - 3.08e-3 * T_AB + 1.51e-4 * T2      (all in us)
         T2 measured from the actual origin.

Tolerances:
  LI 1.2/50 us: T1 +-30%, T2 +-20%, overshoot <= 10%, test voltage +-3%
  SI 250/2500 us: Tp +-20%, T2 +-60%, test voltage +-3%

The 10% overshoot limit and the beta definition follow the 2010 edition.
The 2025 edition redefines the standard switching impulse as 170/2500 us;
confirm which edition CPRI evaluates against before the demo.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

TARGETS = {
    "LI": {"t1": 1.2, "t2": 50.0, "t1_tol": 0.30, "t2_tol": 0.20,
           "overshoot_max": 0.10, "v_tol": 0.03},
    "SI": {"t1": 250.0, "t2": 2500.0, "t1_tol": 0.20, "t2_tol": 0.60,
           "overshoot_max": None, "v_tol": 0.03},
}


@dataclass
class Result:
    kind: str
    test_voltage: float      # V
    extreme: float           # V, raw peak of the record
    t1: float                # us — front time (LI) or time to peak Tp (SI)
    t2: float                # us — time to half value
    overshoot: float         # relative, LI only
    base_curve: np.ndarray = None
    test_curve: np.ndarray = None

    def errors(self, target_v):
        T = TARGETS[self.kind]
        return {
            "t1": (self.t1 - T["t1"]) / T["t1"],
            "t2": (self.t2 - T["t2"]) / T["t2"],
            "v": (self.test_voltage - target_v) / target_v,
            "overshoot": self.overshoot,
        }

    def compliant(self, target_v):
        T, e = TARGETS[self.kind], self.errors(target_v)
        ok = (abs(e["t1"]) <= T["t1_tol"] and abs(e["t2"]) <= T["t2_tol"]
              and abs(e["v"]) <= T["v_tol"])
        if T["overshoot_max"] is not None:
            ok = ok and e["overshoot"] <= T["overshoot_max"]
        return bool(ok)

    def margin(self, target_v, shape_only=False):
        """Worst normalised distance to a tolerance edge. Below 1.0 means
        inside every band; the design search minimises this."""
        T, e = TARGETS[self.kind], self.errors(target_v)
        parts = [abs(e["t1"]) / T["t1_tol"], abs(e["t2"]) / T["t2_tol"]]
        if not shape_only:
            parts.append(abs(e["v"]) / T["v_tol"])
        if T["overshoot_max"] is not None:
            parts.append(e["overshoot"] / T["overshoot_max"])
        return float(max(parts))


# ---------------------------------------------------------------------------
def k_factor(f_mhz):
    """IEC 60060-1:2010 test voltage function."""
    return 1.0 / (1.0 + 2.2 * np.asarray(f_mhz) ** 2)


def k_filter(residual, dt_s):
    """Zero-phase application of the test voltage function."""
    n = len(residual)
    spec = np.fft.rfft(residual)
    f_mhz = np.fft.rfftfreq(n, dt_s) / 1e6
    return np.fft.irfft(spec * k_factor(f_mhz), n)


def _crossings(t, u, level):
    """Interpolated first crossing on the rise and last crossing on the fall."""
    above = u >= level
    if not above.any():
        return np.nan, np.nan
    i = int(np.argmax(above))
    t_rise = t[0] if i == 0 else np.interp(level, [u[i - 1], u[i]],
                                           [t[i - 1], t[i]])
    j = len(u) - 1 - int(np.argmax(above[::-1]))
    t_fall = t[-1] if j >= len(u) - 1 else np.interp(
        level, [u[j + 1], u[j]], [t[j + 1], t[j]])
    return t_rise, t_fall


def _front_tail(t_us, curve, kind):
    peak = curve.max()
    t30, _ = _crossings(t_us, curve, 0.30 * peak)
    t90, _ = _crossings(t_us, curve, 0.90 * peak)
    _, t50 = _crossings(t_us, curve, 0.50 * peak)
    if kind == "LI":
        t1 = 1.67 * (t90 - t30)
        o1 = t30 - 0.3 * t1
        return t1, t50 - o1
    t_ab = t90 - t30
    t2 = t50 - t_us[0]                       # from the actual origin
    k = 2.42 - 3.08e-3 * t_ab + 1.51e-4 * t2
    return k * t_ab, t2


def _double_exp(t, U, t0, tau1, tau2):
    x = np.clip(t - t0, 0, None)
    return U * (np.exp(-x / tau1) - np.exp(-x / tau2))


def fit_base_curve(t_us, u):
    """Least-squares double exponential over the record from the 20% point on
    the front onward. With oscillation present this recovers the mean curve,
    which is what the standard's base curve represents."""
    peak = u.max()
    i20 = int(np.argmax(u >= 0.2 * peak))
    tt, uu = t_us[i20:], u[i20:]

    t_pk = t_us[int(np.argmax(u))]
    p0 = [peak * 1.05, 0.0, max(t_us[-1] / 2.5, 5 * t_pk), max(t_pk / 3, 1e-3)]
    lo = [0.5 * peak, -t_pk, t_pk, 1e-4]
    hi = [3.0 * peak, t_pk, 50 * t_us[-1], 5 * t_pk + 1e-3]
    sol = least_squares(lambda p: _double_exp(tt, *p) - uu, p0,
                        bounds=(lo, hi), x_scale="jac", max_nfev=400)
    return _double_exp(t_us, *sol.x)


def evaluate(t_s, u, kind, base_curve=None):
    """Evaluate a record per IEC 60060-1.

    base_curve may be supplied when it is known exactly (the design model
    can compute the no-inductance response of its own circuit). A measured
    record must be evaluated with the fitted base curve.
    """
    t_us = t_s * 1e6
    extreme = float(u.max())

    if kind == "SI":
        t1, t2 = _front_tail(t_us, u, "SI")
        return Result("SI", extreme, extreme, t1, t2, 0.0, u, u)

    base = fit_base_curve(t_us, u) if base_curve is None else base_curve
    test_curve = base + k_filter(u - base, t_s[1] - t_s[0])
    t1, t2 = _front_tail(t_us, test_curve, "LI")
    overshoot = max(0.0, (extreme - base.max()) / extreme)
    return Result("LI", float(test_curve.max()), extreme, t1, t2,
                  float(overshoot), base, test_curve)


# ---------------------------------------------------------------------------
def _checks():
    ok = True

    def chk(name, got, lo, hi, unit=""):
        nonlocal ok
        p = lo <= got <= hi
        ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}] {name:56s} {got:9.4f}{unit} "
              f"(expect {lo}-{hi})")

    print("Test voltage function")
    chk("k(0) = 1", float(k_factor(0.0)), 0.999, 1.001)
    chk("k(0.5 MHz) = 1/(1+0.55)", float(k_factor(0.5)), 0.644, 0.647)
    chk("k(1 MHz)", float(k_factor(1.0)), 0.311, 0.314)

    t = np.arange(0, 120e-6, 10e-9)
    tu = t * 1e6
    # Canonical 1.2/50 us double exponential
    a, b = 1 / 68.22, 1 / 0.4050
    u = np.exp(-a * tu) - np.exp(-b * tu)
    r = evaluate(t, u, "LI")
    print("\nLightning impulse, smooth 1.2/50 double exponential")
    chk("front time T1", r.t1, 1.17, 1.23, " us")
    chk("time to half T2", r.t2, 49.0, 51.0, " us")
    chk("no overshoot on a smooth wave", r.overshoot, 0.0, 0.01)
    chk("test voltage = extreme when smooth", r.test_voltage / r.extreme,
        0.995, 1.005)

    print("\nLightning impulse with a 1 MHz oscillation on the peak")
    osc = u + 0.12 * u.max() * np.sin(2 * np.pi * 1.0 * tu) * np.exp(-tu / 4.0)
    r2 = evaluate(t, osc, "LI")
    # The true base curve is known here (u), so the true overshoot is
    # computable independently of the fit. Check the fitted base recovers it.
    true_os = (osc.max() - u.max()) / osc.max()
    chk("overshoot recovered by base-curve fit (vs true)",
        r2.overshoot / true_os, 0.80, 1.20)
    chk("test voltage below raw extreme (K-factor attenuates)",
        r2.test_voltage / r2.extreme, 0.85, 0.999)
    chk("front time still recovered", r2.t1, 0.9, 1.5, " us")

    print("\nSwitching impulse")
    ts = np.arange(0, 7000e-6, 2e-6)
    tsu = ts * 1e6
    us_ = np.exp(-tsu / 3650.0) - np.exp(-tsu / 65.0)
    r3 = evaluate(ts, us_, "SI")
    t_peak = tsu[int(np.argmax(us_))]
    chk("IEC Tp formula agrees with the true time to peak",
        r3.t1 / t_peak, 0.95, 1.05)
    chk("time to half in the 2500 us class", r3.t2, 2000, 3000, " us")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _checks() else 1)
