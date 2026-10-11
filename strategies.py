"""
strategies.py — four ways to get from "request" to "compliant waveform".

  S1  technician      Textbook first guess, then the rule of thumb every
                      lab uses: scale Rf by the front-time error, Rt by the
                      tail-time error, bump Rf if it rings, snap to the
                      nearest resistor in stock. This is current practice,
                      and it is a strong baseline — it is what we must beat.

  S2  black-box BO    Bayesian optimisation with a Gaussian process over the
                      available resistors, knowing nothing about circuits.
                      Included to test whether "AI" without physics helps.

  S3  calibration     After each shot, fit the circuit's hidden parameters —
                      loop inductance, load capacitance, divider response,
                      efficiency — to the measured waveform, then redesign
                      with the calibrated model. Physics-informed learning:
                      the waveform itself tells us how this hall differs
                      from the textbook.

  S4  lab memory      S3, but the FIRST shot already uses what this hall has
                      learned from its previous sessions: its parasitics and
                      each insulator type's capacitance bias. The system
                      gets better every time the lab fires.

All four adjust charging voltage by the measured ratio each shot, keep the
stage count fixed after the first shot (re-erecting the generator is real
work), and stop at the first IEC-compliant shot.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares

from circuit import Generator, Setting, simulate
from design import Belief, Correction, design, model_params, predict
from waveform import evaluate, TARGETS

MAX_SHOTS = 10


def _snap(value, stock):
    stock = np.asarray(stock, dtype=float)
    return float(stock[np.argmin(np.abs(np.log(stock) - np.log(value)))])


def _step(value, stock, direction):
    stock = sorted(stock)
    i = stock.index(value)
    return float(stock[int(np.clip(i + direction, 0, len(stock) - 1))])


def _stocks(kind, g):
    return (g.rf_li, g.rt_li) if kind == "LI" else (g.rf_si, g.rt_si)


def _voltage_fix(s, r, target_v):
    return replace(s, v_charge=s.v_charge * target_v / max(r.test_voltage, 1))


def _log(shots, s, r, target_v):
    shots.append({"setting": s, "result": r,
                  "compliant": r.compliant(target_v),
                  "margin": r.margin(target_v)})


# ---------------------------------------------------------------------------
# S1 — current practice
# ---------------------------------------------------------------------------
def technician(lab, obj, seed, g: Generator = None):
    g = g or Generator()
    rf_stock, rt_stock = _stocks(obj.kind, g)
    T = TARGETS[obj.kind]
    s = design(obj.kind, obj.target_v, obj.c_nominal, Belief(), g, top=1)[0]["setting"]
    shots = []
    for _ in range(MAX_SHOTS):
        t, u = lab.fire(s, obj, seed)
        r = evaluate(t, u, obj.kind)
        _log(shots, s, r, obj.target_v)
        if shots[-1]["compliant"]:
            break
        e = r.errors(obj.target_v)
        rf, rt = s.rf_stage, s.rt_stage
        if T["overshoot_max"] and r.overshoot > T["overshoot_max"]:
            rf = _step(rf, rf_stock, +1)                   # damp the ringing
        elif abs(e["t1"]) > T["t1_tol"] * 0.8:
            new = _snap(rf * T["t1"] / r.t1, rf_stock)
            rf = new if new != rf else _step(rf, rf_stock,
                                             -1 if e["t1"] > 0 else +1)
        if abs(e["t2"]) > T["t2_tol"] * 0.8:
            new = _snap(rt * T["t2"] / r.t2, rt_stock)
            rt = new if new != rt else _step(rt, rt_stock,
                                             -1 if e["t2"] > 0 else +1)
        s = _voltage_fix(replace(s, rf_stage=rf, rt_stage=rt), r, obj.target_v)
    return shots


# ---------------------------------------------------------------------------
# S2 — black-box Bayesian optimisation
# ---------------------------------------------------------------------------
def blackbox_bo(lab, obj, seed, g: Generator = None):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from scipy.stats import norm

    g = g or Generator()
    rf_stock, rt_stock = _stocks(obj.kind, g)
    cl_stock = (0.0,) if obj.kind == "LI" else g.cload_si
    s = design(obj.kind, obj.target_v, obj.c_nominal, Belief(), g, top=1)[0]["setting"]

    grid = np.array([(rf, rt, cl) for rf in rf_stock for rt in rt_stock
                     for cl in cl_stock])
    X_all = np.c_[np.log(grid[:, 0]), np.log(grid[:, 1]),
                  np.log(grid[:, 2] + 1e-10)]
    X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-9)

    X, y, shots = [], [], []
    for _ in range(MAX_SHOTS):
        t, u = lab.fire(s, obj, seed)
        r = evaluate(t, u, obj.kind)
        _log(shots, s, r, obj.target_v)
        if shots[-1]["compliant"]:
            break
        idx = int(np.where((grid[:, 0] == s.rf_stage) &
                           (grid[:, 1] == s.rt_stage) &
                           (grid[:, 2] == s.c_load))[0][0])
        X.append(X_all[idx])
        y.append(np.log(r.margin(obj.target_v, shape_only=True) + 1e-3))

        gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(1e-3),
                                      normalize_y=True, random_state=0)
        gp.fit(np.array(X), np.array(y))
        mu, sd = gp.predict(X_all, return_std=True)
        best = min(y)
        z = (best - mu) / (sd + 1e-9)
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        for xi in X:                                    # do not repeat a shot
            ei[np.all(np.isclose(X_all, xi), axis=1)] = -np.inf
        nxt = grid[int(np.argmax(ei))]
        s = _voltage_fix(replace(s, rf_stage=float(nxt[0]),
                                 rt_stage=float(nxt[1]), c_load=float(nxt[2])),
                         r, obj.target_v)
    return shots


# ---------------------------------------------------------------------------
# S3 — physics-informed calibration
# ---------------------------------------------------------------------------
def calibrate_joint(history, obj, g, prior: Belief):
    """Fit the hall's hidden parameters to EVERY shot so far in the session.

    From a single waveform, loop inductance and divider response are
    confounded — both slow the front. Shots at different resistor settings
    excite the circuit differently, which separates them. This is ordinary
    system identification: more varied excitation, better identifiability.
    """
    if len(history) == 1:
        s, t, u = history[0]
        return calibrate(s, obj, t, u, g, prior)
    s0 = history[-1][0]
    p_last = model_params(s0, g, obj.c_nominal, prior)

    def resid(x):
        lf, c2, tm, gain = np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), x[3]
        out = []
        for s, t, u in history:
            p = model_params(s, g, obj.c_nominal, Belief())
            l_nom = s.n * g.l_stage + g.l_ext
            pp = replace(p, l=lf * l_nom, c2=c2, tm=tm)
            out.append((gain * simulate(pp, t) - u) / u.max())
        return np.concatenate(out)

    l_nom0 = s0.n * g.l_stage + g.l_ext
    x0 = [np.log(max(prior.l_factor, 0.3)), np.log(p_last.c2),
          np.log(max(prior.tm, 20e-9)), float(prior.eff_scale)]
    lo = [np.log(0.3), np.log(p_last.c2 * 0.3), np.log(1e-9), 0.8]
    hi = [np.log(8.0), np.log(p_last.c2 * 4.0), np.log(400e-9), 1.2]
    x0 = list(np.clip(x0, lo, hi))
    sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=150)
    lf, c2, tm, gain = np.exp(sol.x[0]), np.exp(sol.x[1]), \
        np.exp(sol.x[2]), sol.x[3]
    c2_model = obj.c_nominal + g.c_fixed + g.c_stray_nominal + s0.c_load
    return Belief(l_factor=float(lf), c2_offset=float(c2 - c2_model),
                  tm=float(tm), eff_scale=float(gain)), \
        {"l": lf * l_nom0, "c2": c2, "tm": tm, "gain": gain}


def calibrate(s, obj, t, u, g, prior: Belief):
    """Fit hidden circuit parameters to one measured record.

    Returns a Belief for this hall. Four parameters: loop inductance, load
    capacitance, divider time constant, and an efficiency gain that absorbs
    resistor and capacitor tolerances. Few enough to be identifiable from a
    single waveform, which is the point — each shot teaches the model.
    """
    p0 = model_params(s, g, obj.c_nominal, prior)
    peak = u.max()
    w = slice(0, len(u))

    def resid(x):
        l, c2, tm, gain = np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), x[3]
        pp = replace(p0, l=l, c2=c2, tm=tm)
        return (gain * simulate(pp, t)[w] - u[w]) / peak

    x0 = [np.log(p0.l), np.log(p0.c2), np.log(max(prior.tm, 20e-9)), 1.0]
    lo = [np.log(p0.l * 0.3), np.log(p0.c2 * 0.3), np.log(1e-9), 0.8]
    hi = [np.log(p0.l * 6.0), np.log(p0.c2 * 6.0), np.log(400e-9), 1.2]
    sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=120)
    l_fit, c2_fit, tm_fit, gain = np.exp(sol.x[0]), np.exp(sol.x[1]), \
        np.exp(sol.x[2]), sol.x[3]

    l_nominal = s.n * g.l_stage + g.l_ext
    c2_model = obj.c_nominal + g.c_fixed + g.c_stray_nominal + s.c_load
    return Belief(l_factor=float(l_fit / l_nominal), l_extra=0.0,
                  c2_offset=float(c2_fit - c2_model), c_test_scale=1.0,
                  tm=float(tm_fit), eff_scale=float(gain)), \
        {"l": l_fit, "c2": c2_fit, "tm": tm_fit, "gain": gain}


def _correction(history_results, obj, g, b):
    """Residual between what the calibrated model predicts at the fired
    settings and what was measured, averaged over recent shots."""
    t1r, t2r, osd = [], [], []
    for s, r in history_results[-3:]:
        p = predict(s, g, obj.c_nominal, b, obj.kind)[0]
        t1r.append(r.t1 / max(p.t1, 1e-9))
        t2r.append(r.t2 / max(p.t2, 1e-9))
        osd.append(r.overshoot - p.overshoot)
    return Correction(float(np.median(t1r)), float(np.median(t2r)),
                      float(np.median(osd)))


COMFORTABLE = 0.8   # fixed BEFORE benchmarking, not tuned on it: the same
                    # 0.8-of-tolerance trigger the technician's rule uses


def _fewest_climbs(cand, cur):
    """Of the settings predicted comfortably inside every band, the one that
    changes fewest resistors from the current setting — each changed
    resistor is a climb up the generator, CPRI's stated pain point — then
    the best-ranked. If none is comfortable, the best-ranked, as before."""
    ok = [c for c in cand if c["shape_margin"] <= COMFORTABLE]
    if not ok:
        return cand[0]["setting"]

    def climbs(c):
        return ((c["setting"].rf_stage != cur.rf_stage) +
                (c["setting"].rt_stage != cur.rt_stage))
    return min(ok, key=lambda c: (climbs(c), c["score"]))["setting"]


def calibrated(lab, obj, seed, g: Generator = None, prior: Belief = None,
               use_correction=True):
    g = g or Generator()
    b = prior or Belief()
    s = design(obj.kind, obj.target_v, obj.c_nominal, b, g, top=1)[0]["setting"]
    n_fixed = s.n
    shots, fits, history = [], [], []
    for _ in range(MAX_SHOTS):
        t, u = lab.fire(s, obj, seed)
        r = evaluate(t, u, obj.kind)
        _log(shots, s, r, obj.target_v)
        history.append((s, t, u))
        b, fit = calibrate_joint(history, obj, g, b)
        fits.append(fit)
        if shots[-1]["compliant"]:
            break
        # Shape in band, only the crest missed: fix the charging voltage at
        # the control desk instead of climbing the generator. The circuit is
        # linear, so the measured ratio is exact up to noise. This used to be
        # missing — any failed shot banned its resistors, forcing a climb
        # even when the waveform shape had already passed.
        last = shots[-1]["result"]
        if last.margin(obj.target_v, shape_only=True) <= 1.0:
            v_fix = _voltage_fix(s, last, obj.target_v)
            if v_fix.v_charge <= g.charge_margin * g.v_stage_max:
                s = v_fix
                continue
        # Never repeat a setting whose SHAPE failed: the model believing it
        # should pass is not evidence against a measurement that it did not.
        tried = {(x["setting"].rf_stage, x["setting"].rt_stage,
                  x["setting"].c_load) for x in shots
                 if x["result"].margin(obj.target_v, shape_only=True) > 1.0}
        corr = _correction([(x["setting"], x["result"]) for x in shots],
                           obj, g, b) if use_correction else None
        cand = design(obj.kind, obj.target_v, obj.c_nominal, b, g, top=40,
                      restrict=lambda n, rf, rt, cl: n == n_fixed and
                      (rf, rt, cl) not in tried, corr=corr)
        if not cand:
            break
        # the model's charging voltage already includes the calibrated gain
        s = _fewest_climbs(cand, s)
    shots[-1]["fits"] = fits
    return shots


# ---------------------------------------------------------------------------
# S5 — the offline app's shot loop
# ---------------------------------------------------------------------------
def app_loop(lab, obj, seed, g: Generator = None, first: Setting = None):
    """What the offline app's shot loop does — benchmarked here before the app
    claims anything for it.

    The browser has no circuit solver, so it cannot fit the hall's parameters
    to the recorded waveform the way `calibrated` does. It has only the four
    numbers an operator reads off the IEC evaluator after a shot: T1 (Tp),
    T2, test voltage and overshoot. So, from the technician's own textbook
    first shot:
      - shape in band, only the crest off: same resistors, charge scaled by
        the measured ratio — no climb;
      - shape out: every candidate's predicted times scaled by the median
        measured/predicted ratio of recent shots (design.Correction), its
        charge by the measured/predicted test-voltage ratio, and of those
        predicted comfortably in band, the one changing fewest resistors.
    Same first shot as `technician`, so any difference is the follow-up —
    unless `first` is given (app_memory's first shot)."""
    g = g or Generator()
    b = Belief()
    s = first or design(obj.kind, obj.target_v, obj.c_nominal, b, g,
                        top=1)[0]["setting"]
    n_fixed = s.n
    shots = []
    for _ in range(MAX_SHOTS):
        t, u = lab.fire(s, obj, seed)
        r = evaluate(t, u, obj.kind)
        _log(shots, s, r, obj.target_v)
        if shots[-1]["compliant"]:
            break
        if r.margin(obj.target_v, shape_only=True) <= 1.0:
            v_fix = _voltage_fix(s, r, obj.target_v)
            if v_fix.v_charge <= g.charge_margin * g.v_stage_max:
                s = v_fix
                continue
        recent = [(x["setting"], x["result"]) for x in shots][-3:]
        tried = {(x["setting"].rf_stage, x["setting"].rt_stage,
                  x["setting"].c_load) for x in shots
                 if x["result"].margin(obj.target_v, shape_only=True) > 1.0}
        corr = _correction(recent, obj, g, b)
        # measured efficiency against the textbook's, at the settings fired
        gain = float(np.median([
            predict(xs, g, obj.c_nominal, b, obj.kind)[0].test_voltage /
            max(xr.test_voltage, 1.0) for xs, xr in recent]))
        cand = design(obj.kind, obj.target_v, obj.c_nominal, b, g, top=40,
                      restrict=lambda n, rf, rt, cl: n == n_fixed and
                      (rf, rt, cl) not in tried, corr=corr)
        if not cand:
            break
        nxt = _fewest_climbs(cand, s)
        s = replace(nxt, v_charge=min(nxt.v_charge * gain,
                                      g.charge_margin * g.v_stage_max))
    return shots


# ---------------------------------------------------------------------------
# S4 — lab memory
# ---------------------------------------------------------------------------
class LabMemory:
    """What a hall learns about itself across sessions.

    Hall constants (loop inductance factor, divider response, efficiency)
    are learned as robust medians. Load capacitance is learned as a
    regression per insulator type — c2 = a_type * c_estimate + b — because
    the estimate error is systematic by type, which a median would miss.
    """

    def __init__(self):
        self.rows = []

    def add(self, obj, s, fit, g):
        self.rows.append({
            "category": obj.category, "c_nom": obj.c_nominal,
            "c2_fit": fit["c2"], "c_load": s.c_load,
            "l_factor": fit["l"] / (s.n * g.l_stage + g.l_ext),
            "tm": fit["tm"], "gain": fit["gain"]})

    def prior(self, obj, g, k_shrink=6.0):
        """Learned belief, shrunk toward the textbook by how much evidence
        there is. With few sessions a learned correction is mostly noise, and
        applying it in a hall where the textbook already works makes things
        worse — the first benchmark showed exactly that (83% vs 100%)."""
        if len(self.rows) < 3:
            return Belief()
        import pandas as pd
        d = pd.DataFrame(self.rows)
        d["c_obj"] = d.c2_fit - d.c_load - g.c_fixed   # object + strays (base load and divider removed)
        n = len(d)
        w = n / (n + k_shrink)

        def shrink(learned, textbook):
            return float(textbook + w * (learned - textbook))

        b = Belief(l_factor=shrink(d.l_factor.median(), 1.0),
                   tm=shrink(d.tm.median(), 0.0),
                   eff_scale=shrink(d.gain.median(), 1.0))

        # Load capacitance: a robust per-type ratio (object+strays)/estimate,
        # not a regression — three points do not support a slope.
        sub = d[d.category == obj.category]
        src = sub if len(sub) >= 2 else d
        wc = len(src) / (len(src) + k_shrink)
        ratio_text = (obj.c_nominal + g.c_stray_nominal) / obj.c_nominal
        ratio_learn = float((src.c_obj / src.c_nom).median())
        ratio = ratio_text + wc * (ratio_learn - ratio_text)
        c2_model = obj.c_nominal + g.c_fixed + g.c_stray_nominal
        b.c2_offset = ratio * obj.c_nominal + g.c_fixed - c2_model
        return b


def with_memory(lab, obj, seed, memory: LabMemory, g: Generator = None):
    g = g or Generator()
    shots = calibrated(lab, obj, seed, g, prior=memory.prior(obj, g))
    return shots


# ---------------------------------------------------------------------------
# S6 — the app's lab memory (four readings per shot, kept in the browser)
# ---------------------------------------------------------------------------
class AppMemory:
    """What the offline app can remember about a hall across tests. It has
    no circuit solver, so it cannot fit parasitics the way LabMemory does;
    it keeps, per test, the median measured/textbook ratio of the four
    evaluator readings over that test's shots. The browser stores exactly
    these records (app_template.html, memoryCorrection).

    The next test's first shot applies the median over past tests of the
    same impulse type — the same insulator type too once there are two —
    shrunk toward the textbook by n/(n + K_SHRINK), and nothing below
    MIN_TESTS. Both constants are LabMemory's, not tuned here."""
    MIN_TESTS = 3
    K_SHRINK = 6.0

    def __init__(self):
        self.tests = []

    def add(self, obj, shots, g):
        rat = []
        for x in shots:
            s, r = x["setting"], x["result"]
            p = predict(s, g, obj.c_nominal, Belief(), obj.kind)[0]
            rat.append((r.t1 / p.t1, r.t2 / p.t2, r.overshoot - p.overshoot,
                        p.test_voltage / max(r.test_voltage, 1.0)))
        m = np.median(np.array(rat), axis=0)
        self.tests.append({"kind": obj.kind, "category": obj.category,
                           "t1": float(m[0]), "t2": float(m[1]),
                           "os": float(m[2]), "gain": float(m[3])})

    def correction(self, kind, category):
        """(Correction, gain, tests used), or None below MIN_TESTS."""
        same = [t for t in self.tests if t["kind"] == kind]
        if len(same) < self.MIN_TESTS:
            return None
        cat = [t for t in same if t["category"] == category]
        src = cat if len(cat) >= 2 else same
        w = len(src) / (len(src) + self.K_SHRINK)
        med = {k: float(np.median([t[k] for t in src]))
               for k in ("t1", "t2", "os", "gain")}
        return (Correction(1 + w * (med["t1"] - 1), 1 + w * (med["t2"] - 1),
                           w * med["os"]),
                1 + w * (med["gain"] - 1), len(src))


def app_memory(lab, obj, seed, memory: AppMemory, g: Generator = None):
    """The app with its hall memory: the first shot is designed with the
    remembered correction (times scaled, charge scaled by the remembered
    efficiency ratio), then the app loop's rules. With too little memory it
    is exactly app_loop."""
    g = g or Generator()
    c = memory.correction(obj.kind, obj.category)
    first = None
    if c is not None:
        corr, gain, _ = c
        cand = design(obj.kind, obj.target_v, obj.c_nominal,
                      Belief(eff_scale=1.0 / gain), g, top=1, corr=corr)
        first = cand[0]["setting"] if cand else None
    return app_loop(lab, obj, seed, g, first=first)
