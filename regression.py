"""
regression.py — answer Dr. Sheshu's question directly.

CPRI's existing tool: physics equations -> WEIGHTED K-NEAREST-NEIGHBOURS
RESIDUAL REGRESSION to correct the gap between theoretical and measured
waveform -> compliance check -> recommend resistors. Students are asked to
make it more accurate, judged by regression metrics (RMSE, MAE, MAPE) and
compliance accuracy.

So this file builds his setup and compares against it:

  * dataset — shots fired in ONE hidden hall (CPRI has one generator, so
    the correction is lab-specific), with random stocked settings and random
    insulators. Each row: what the operator knows, what the textbook
    predicts, what the hall actually produced.
  * models — his weighted KNN as the baseline, plus linear, SVR, random
    forest, gradient boosting, a Gaussian process, and PHYSICS CALIBRATION:
    fit a handful of hidden circuit parameters to the training shots and let
    the physics do the rest.
  * two tests — random cross-validation (what a residual regressor is
    usually scored on) and an EXTRAPOLATION split, training on smaller
    insulators and testing on larger ones. A real lab keeps meeting setups
    it has not seen; that is where averaging-based methods like KNN fail.

    python regression.py --n 300          # build dataset + evaluate
    python regression.py --reuse          # evaluate an existing dataset
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

from circuit import Generator, Setting
from waveform import evaluate, TARGETS
from lab import hall, make_session
from design import Belief, predict, candidates

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")
DATA = os.path.join(RESULTS, "regression_dataset.csv")
TARGET_COLS = ["t1", "t2", "eff"]   # front time, tail time, and "eff" =
                                     # test voltage / charging voltage (can
                                     # exceed 1 when the circuit rings)
FEATURES = ["is_si", "n", "v_charge_kv", "rf_stage", "rt_stage", "c_load_nf",
            "c_test_pf", "th_t1", "th_t2", "th_eff"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(n_rows=300, hall_seed=4242, severity=3.0, seed=1):
    """Shots in one hidden hall. Theory = textbook belief, same IEC evaluator
    as the lab, so the residual is purely the hall's departure from theory."""
    g = Generator()
    lab = hall(hall_seed, severity)
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i in range(n_rows):
        obj = make_session(rng)
        cands = list(candidates(obj.kind, g))
        T = TARGETS[obj.kind]
        # A real shot log only holds settings an engineer thought were close,
        # so sample the way one would: keep settings the textbook places in
        # the neighbourhood of the target. Random settings (0.2 us fronts for
        # a 1.2 us target) are shots nobody fires, and would distort every
        # model's score in ways that say nothing about a real hall.
        for _ in range(60):
            n, rf, rt, cl = cands[int(rng.integers(len(cands)))]
            s = Setting(n=n, v_charge=float(rng.uniform(.35, .90)) *
                        g.v_stage_max, rf_stage=rf, rt_stage=rt, c_load=cl)
            th, _, _ = predict(s, g, obj.c_nominal, Belief(), obj.kind)
            if (np.isfinite(th.t1) and .55 <= th.t1 / T["t1"] <= 1.8 and
                    .60 <= th.t2 / T["t2"] <= 1.6):
                break
        else:
            continue
        t, u = lab.fire(s, obj, session_seed=seed * 10_000 + i)
        pr = evaluate(t, u, obj.kind)
        if not (np.isfinite(th.t1) and np.isfinite(pr.t1)):
            continue
        v0 = n * s.v_charge
        rows.append({
            "kind": obj.kind, "category": obj.category, "is_si": int(obj.kind == "SI"),
            "n": n, "v_charge_kv": s.v_charge / 1e3, "rf_stage": rf, "rt_stage": rt,
            "c_load_nf": cl * 1e9, "c_test_pf": obj.c_nominal * 1e12,
            "th_t1": th.t1, "th_t2": th.t2, "th_eff": abs(th.test_voltage) / v0,
            "t1": pr.t1, "t2": pr.t2, "eff": abs(pr.test_voltage) / v0,
            "v0_kv": v0 / 1e3,
            # overshoot (P8): appended last so the RNG stream, and every
            # column above, is unchanged from the log built without it
            "th_os": th.overshoot, "os": pr.overshoot,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_rows} shots  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models — each predicts the RESIDUAL (actual - theory), as in CPRI's design
# ---------------------------------------------------------------------------
def make_mlp(seed=0):
    """A neural network, set up the way networks are fairly compared on
    small tables: scaled inputs AND targets, weight decay, early stopping.
    One fixed seed, like every other model here — but networks are
    seed-sensitive, so CLAUDE.md reports the spread over five seeds too."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import TransformedTargetRegressor
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=4000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=60, random_state=seed)),
        transformer=StandardScaler())


def make_models():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.multioutput import MultiOutputRegressor

    sc = StandardScaler
    return {
        "Weighted KNN (CPRI baseline)": lambda: make_pipeline(
            sc(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "Ridge": lambda: make_pipeline(sc(), Ridge(alpha=1.0)),
        "SVR": lambda: make_pipeline(sc(), MultiOutputRegressor(
            SVR(C=3.0, epsilon=0.01))),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "Gradient boosting": lambda: MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=0)),
        "Gaussian process": lambda: make_pipeline(sc(), GaussianProcessRegressor(
            ConstantKernel() * RBF(np.ones(len(FEATURES))) + WhiteKernel(1e-2),
            normalize_y=True, n_restarts_optimizer=1, random_state=0)),
        "Neural network (MLP 64-64)": make_mlp,
    }


def residual_targets(df):
    """Relative residuals, so front (µs) and tail (tens of µs) are comparable."""
    return np.column_stack([df.t1 / df.th_t1 - 1, df.t2 / df.th_t2 - 1,
                            df.eff / df.th_eff - 1])


def apply_residual(df, r):
    return np.column_stack([df.th_t1 * (1 + r[:, 0]), df.th_t2 * (1 + r[:, 1]),
                            df.th_eff * (1 + r[:, 2])])


# ---------------------------------------------------------------------------
# Physics calibration — few parameters, physics carries the structure
# ---------------------------------------------------------------------------
def calibrate_hall(train, max_shots=15, seed=0):
    """Fit loop inductance, extra load capacitance, test-object scale and an
    efficiency factor to training shots by least squares on the IEC outputs.
    Four numbers describe the hall instead of a table of examples, which is
    why this can extrapolate where KNN cannot."""
    from scipy.optimize import least_squares
    g = Generator()
    sub = train.sample(min(max_shots, len(train)), random_state=seed)

    def resid(x):
        b = Belief(l_factor=x[0], l_extra=x[1] * 1e-6, c2_offset=x[2] * 1e-12,
                   tm=x[3] * 1e-9, eff_scale=1.0)
        out = []
        for r in sub.itertuples():
            s = Setting(int(r.n), r.v_charge_kv * 1e3, r.rf_stage, r.rt_stage,
                        r.c_load_nf * 1e-9)
            pr, _, _ = predict(s, g, r.c_test_pf * 1e-12, b, r.kind)
            out += [pr.t1 / r.t1 - 1, pr.t2 / r.t2 - 1]
        return np.nan_to_num(np.array(out), nan=1.0)

    # The divider response time tm was missing from the first version, and
    # the misspecified model extrapolated worse than KNN (10.9% front-time
    # error). A wrong physics model does not fail gracefully: it confidently
    # extrapolates the wrong thing.
    # c_test_scale was dropped: it and c2_offset both just add capacitance,
    # so on small insulators the fit cannot tell them apart, and on large
    # ones the scale term amplifies whichever split it chose.
    x = least_squares(resid, [1.0, 5.0, 100.0, 20.0],
                      bounds=([.5, 0, 0, 0], [4, 60, 800, 250]),
                      diff_step=0.05, max_nfev=25).x
    b = Belief(l_factor=x[0], l_extra=x[1] * 1e-6, c2_offset=x[2] * 1e-12,
               tm=x[3] * 1e-9)
    # efficiency: one multiplicative factor, fitted in closed form
    eff_pred = []
    for r in train.itertuples():
        s = Setting(int(r.n), r.v_charge_kv * 1e3, r.rf_stage, r.rt_stage,
                    r.c_load_nf * 1e-9)
        pr, _, _ = predict(s, g, r.c_test_pf * 1e-12, b, r.kind)
        eff_pred.append(abs(pr.test_voltage) / (r.v0_kv * 1e3))
    b.eff_scale = float(np.median(train.eff.to_numpy() / np.array(eff_pred)))
    return b


def physics_predict(b, df, overshoot=False):
    """Calibrated physics [t1, t2, eff], plus overshoot as a fourth column
    when asked (a fraction, as waveform.Result reports it)."""
    g = Generator()
    out = []
    for r in df.itertuples():
        s = Setting(int(r.n), r.v_charge_kv * 1e3, r.rf_stage, r.rt_stage,
                    r.c_load_nf * 1e-9)
        pr, _, _ = predict(s, g, r.c_test_pf * 1e-12, b, r.kind)
        out.append([pr.t1, pr.t2,
                    b.eff_scale * abs(pr.test_voltage) / (r.v0_kv * 1e3)]
                   + ([pr.overshoot] if overshoot else []))
    return np.array(out)


# ---------------------------------------------------------------------------
# Scoring — the metrics Dr. Sheshu named
# ---------------------------------------------------------------------------
def compliant(df, t1, t2):
    ok = np.zeros(len(df), bool)
    for kind in ("LI", "SI"):
        T = TARGETS[kind]
        m = (df.kind == kind).to_numpy()
        ok[m] = (np.abs(t1[m] / T["t1"] - 1) <= T["t1_tol"]) & \
                (np.abs(t2[m] / T["t2"] - 1) <= T["t2_tol"])
    return ok


def score(df, pred, name, split):
    act = df[TARGET_COLS].to_numpy()
    row = {"split": split, "model": name}
    for j, c in enumerate(TARGET_COLS):
        e = pred[:, j] - act[:, j]
        row[f"{c}_mape_%"] = round(100 * np.mean(np.abs(e / act[:, j])), 2)
        row[f"{c}_rmse"] = round(float(np.sqrt(np.mean(e ** 2))), 4)
    # would the operator be told the right pass/fail on waveshape?
    row["compliance_acc_%"] = round(100 * np.mean(
        compliant(df, pred[:, 0], pred[:, 1]) ==
        compliant(df, act[:, 0], act[:, 1])), 1)
    return row


# ---------------------------------------------------------------------------
# The hybrid — one definition, used by the benchmark and shipped to the app
# ---------------------------------------------------------------------------
HYBRID_FEATURES = FEATURES + ["ph_t1", "ph_t2", "ph_eff"]


def _hybrid_gb():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0))


def fit_hybrid(train, learner=None, b=None):
    """Physics calibration carries the structure; a learner (gradient
    boosting by default — the shipped hybrid) fits what physics still misses
    (per-resistor tolerance, per-insulator-type bias), as a relative residual
    on the calibrated prediction. Pass `b` to reuse a calibration."""
    if b is None:
        b = calibrate_hall(train)
    ph = physics_predict(b, train)
    m = (learner or _hybrid_gb)()
    m.fit(np.column_stack([train[FEATURES].to_numpy(), ph]),
          train[TARGET_COLS].to_numpy() / ph - 1)
    return b, m


def hybrid_predict(b, gb, df):
    """(calibrated physics prediction, hybrid prediction) for t1, t2, eff."""
    ph = physics_predict(b, df)
    corr = gb.predict(np.column_stack([df[FEATURES].to_numpy(), ph]))
    return ph, ph * (1 + corr)


# ---------------------------------------------------------------------------
# Overshoot (P8) — a fourth corrected quantity, lightning only
# ---------------------------------------------------------------------------
# Separate from the three-target hybrid on purpose: adding a column to the
# multi-output models would move every number already reported for them.
# Additive residual in overshoot (a fraction), not relative: physics often
# predicts ~0 overshoot where the hall rings, so a ratio is meaningless.
OS_FEATURES = FEATURES + ["th_os"]
RINGING = 0.05     # "ringing risk" flag: half the IEC 60060-1 10% limit.
                   # Fixed by the limit, not tuned on the log.


def _os_gb():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, random_state=0)


def fit_overshoot(train, b):
    """Boosted trees on (measured - calibrated physics) overshoot, LI shots
    only (IEC sets no overshoot limit for switching). Inputs are the
    hybrid's features plus the calibrated physics overshoot; reuses the
    hybrid's calibration `b`."""
    li = train[train.kind == "LI"]
    ph = physics_predict(b, li, overshoot=True)
    return _os_gb().fit(np.column_stack([li[FEATURES].to_numpy(), ph]),
                        li.os.to_numpy() - ph[:, 3])


def overshoot_predict(b, m, df):
    """(calibrated physics overshoot, hybrid overshoot), never below zero."""
    ph = physics_predict(b, df, overshoot=True)
    corr = m.predict(np.column_stack([df[FEATURES].to_numpy(), ph]))
    return ph[:, 3], np.maximum(0.0, ph[:, 3] + corr)


def _os_score(test, pred, name, split):
    act = test.os.to_numpy()
    e = 100 * (pred - act)                                 # percentage points
    return {"split": split, "model": name, "shots": len(test),
            "os_mae_pp": round(float(np.abs(e).mean()), 2),
            "os_rmse_pp": round(float(np.sqrt((e ** 2).mean())), 2),
            "os_max_err_pp": round(float(np.abs(e).max()), 2),
            "ringing_flag_acc_%": round(100 * float(np.mean(
                (pred >= RINGING) == (act >= RINGING))), 1),
            "actual_ringing_%": round(100 * float(np.mean(act >= RINGING)), 1)}


def _os_split(train, test, split):
    tr, te = train[train.kind == "LI"], test[test.kind == "LI"]
    th = te.th_os.to_numpy()
    rows = [_os_score(te, th, "Textbook only (no ML)", split)]
    Xtr, Xte = tr[OS_FEATURES].to_numpy(), te[OS_FEATURES].to_numpy()
    y = tr.os.to_numpy() - tr.th_os.to_numpy()
    for name, mk in (("Weighted KNN (CPRI baseline)",
                      make_models()["Weighted KNN (CPRI baseline)"]),
                     ("Gradient boosting", _os_gb)):
        m = mk().fit(Xtr, y)
        rows.append(_os_score(te, np.maximum(0, th + m.predict(Xte)), name, split))
    b = calibrate_hall(train)
    ph, hy = overshoot_predict(b, fit_overshoot(train, b), te)
    rows.append(_os_score(te, ph, "Physics calibration", split))
    rows.append(_os_score(te, hy, "Hybrid: physics + boosting (ours)", split))
    return rows


def overshoot_eval(df, folds=3, seed=0):
    """Same folds and extrapolation cut as evaluate_all, scored on the
    lightning shots of each test fold."""
    rows = []
    idx = np.random.default_rng(seed).permutation(len(df))
    parts = np.array_split(idx, folds)
    for k in range(folds):
        te = df.iloc[parts[k]]
        tr = df.iloc[np.concatenate([parts[j] for j in range(folds) if j != k])]
        rows += _os_split(tr, te, "random CV")
    cv = pd.DataFrame(rows).groupby(["split", "model"], sort=False).mean(
        numeric_only=True).round(2).reset_index()
    cut = df.c_test_pf.quantile(0.70)
    ex = pd.DataFrame(_os_split(df[df.c_test_pf <= cut], df[df.c_test_pf > cut],
                                "extrapolation: larger insulators"))
    return pd.concat([cv, ex], ignore_index=True)


# ---------------------------------------------------------------------------
# Similar past shots — the evidence behind a prediction, as CPRI's kNN shows
# ---------------------------------------------------------------------------
SIMILAR_K = 5      # the CPRI-baseline KNN's own k (make_models)


def similar_shots(log, query, k=SIMILAR_K):
    """The k logged shots nearest one query setting, and what the CPRI-style
    KNN predicts from them: make_models' baseline (StandardScaler, then
    distance-weighted KNeighborsRegressor(5) on relative residuals), fitted
    on the query's impulse type only. Returns (log index labels, [t1, t2,
    eff]). The browser's similarShots must reproduce this."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    sub = log[log.kind == query.kind.iloc[0]]
    sc = StandardScaler().fit(sub[FEATURES].to_numpy())
    Xs, Xq = sc.transform(sub[FEATURES].to_numpy()), sc.transform(query[FEATURES].to_numpy())
    knn = KNeighborsRegressor(k, weights="distance").fit(Xs, residual_targets(sub))
    _, nb = knn.kneighbors(Xq)
    return sub.index.to_numpy()[nb[0]], apply_residual(query, knn.predict(Xq))[0]


def gb_to_json(gb):
    """One fitted GradientBoostingRegressor as plain arrays the browser can
    walk. Prediction = init + lr * sum(leaf value of each tree). Trees
    compare float32(x) <= threshold, as sklearn does (it casts X to float32
    before descending), so the browser must round with Math.fround."""
    assert gb.loss == "squared_error" and hasattr(gb.init_, "constant_")
    trees = []
    for est in gb.estimators_[:, 0]:
        t = est.tree_
        leaf = t.children_left == -1
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "f": np.where(leaf, 0, t.feature).tolist(),
                      "th": np.where(leaf, 0.0, t.threshold).tolist(),
                      "v": np.where(leaf, t.value[:, 0, 0], 0.0).tolist()})
    return {"init": float(gb.init_.constant_.ravel()[0]),
            "lr": float(gb.learning_rate), "trees": trees}


def gb_json_predict(m, X):
    """Walk gb_to_json() output exactly as the browser does. Used to prove
    the export loses nothing before the page is ever opened."""
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    out = np.full(len(X), m["init"])
    for t in m["trees"]:
        for k, x in enumerate(X):
            i = 0
            while t["l"][i] != -1:
                i = t["l"][i] if x[t["f"][i]] <= t["th"][i] else t["r"][i]
            out[k] += m["lr"] * t["v"][i]
    return out


def run_split(train, test, split_name):
    rows = []
    th = test[["th_t1", "th_t2", "th_eff"]].to_numpy()
    rows.append(score(test, th, "Textbook only (no ML)", split_name))
    Xtr, Xte = train[FEATURES].to_numpy(), test[FEATURES].to_numpy()
    ytr = residual_targets(train)
    for name, mk in make_models().items():
        m = mk().fit(Xtr, ytr)
        rows.append(score(test, apply_residual(test, m.predict(Xte)),
                          name, split_name))
    b, gb = fit_hybrid(train)
    ph_te, hy_te = hybrid_predict(b, gb, test)
    rows.append(score(test, ph_te, "Physics calibration", split_name))
    rows.append(score(test, hy_te, "Hybrid: physics + boosting (ours)",
                      split_name))
    # Same hybrid, a neural network in place of the trees — same calibration.
    _, nn = fit_hybrid(train, learner=make_mlp, b=b)
    rows.append(score(test, hybrid_predict(b, nn, test)[1],
                      "Hybrid: physics + MLP", split_name))
    return rows


def evaluate_all(df, folds=3, seed=0):
    rows = []
    idx = np.random.default_rng(seed).permutation(len(df))
    parts = np.array_split(idx, folds)
    for k in range(folds):
        te = df.iloc[parts[k]]
        tr = df.iloc[np.concatenate([parts[j] for j in range(folds) if j != k])]
        rows += run_split(tr, te, "random CV")
    cv = pd.DataFrame(rows).groupby(["split", "model"], sort=False).mean(
        numeric_only=True).round(2).reset_index()

    # Extrapolation: train on the smaller 70% of insulators, test on larger
    cut = df.c_test_pf.quantile(0.70)
    ex = pd.DataFrame(run_split(df[df.c_test_pf <= cut], df[df.c_test_pf > cut],
                                "extrapolation: larger insulators"))
    return pd.concat([cv, ex], ignore_index=True)


def learning_curve(df, sizes=(20, 40, 80, 160, 250), test_n=100, seed=0):
    """How many past shots does each method need? A lab's history for any
    one setup is limited, so the few-shot regime matters most."""
    idx = np.random.default_rng(seed).permutation(len(df))
    test = df.iloc[idx[:test_n]]
    pool = df.iloc[idx[test_n:]]
    Xte = test[FEATURES].to_numpy()
    rows = []
    for n in sizes:
        tr = pool.iloc[:n]
        Xtr, ytr = tr[FEATURES].to_numpy(), residual_targets(tr)
        for name in ("Weighted KNN (CPRI baseline)", "Gradient boosting"):
            m = make_models()[name]().fit(Xtr, ytr)
            r = score(test, apply_residual(test, m.predict(Xte)), name, n)
            rows.append(r)
        b, gb = fit_hybrid(tr)
        ph_te, hy_te = hybrid_predict(b, gb, test)
        rows.append(score(test, hy_te, "Hybrid: physics + boosting (ours)", n))
        rows.append(score(test, ph_te, "Physics calibration", n))
        print(f"  {n} shots done", flush=True)
    out = pd.DataFrame(rows).rename(columns={"split": "train_shots"})
    out.to_csv(os.path.join(RESULTS, "regression_learning_curve.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# CPRI's own dataset — their weighted kNN, reproduced exactly, on their data
# ---------------------------------------------------------------------------
CPRI_XLSX = os.path.join(HERE, "data", "cpri_optimiser.xlsx")
CPRI_X = ["si", "Test_kV", "Load_C_pF", "Divider_C_pF", "Stray_C_pF", "L_uH",
          "Efficiency", "Front_R_Stage", "Tail_R_Stage"]
CPRI_PH = ["Physics_FrontPeak_us", "Physics_Tail_us", "Physics_Crest_kV"]
CPRI_OB = ["Observed_FrontPeak_us", "Observed_Tail_us", "Observed_Crest_kV"]
CPRI_RS = ["Residual_FrontPeak", "Residual_Tail", "Residual_Crest"]
# What CPRI's "Model Validation" sheet reports for its kNN: MAE, RMSE, MAPE %
# for front/peak, tail, crest. Our reimplementation must reproduce these.
CPRI_REPORTED = {"mae": (0.8029, 12.3111, 4.0958),
                 "rmse": (1.3269, 20.3673, 4.9529),
                 "mape": (0.6088, 1.0555, 0.3326)}


def load_cpri(path=CPRI_XLSX):
    """The 'Synthetic Dataset' sheet. CPRI scores on Validation + Hidden Test
    together after training on Train — established by reproducing their
    reported metrics to four decimals, which no other split does."""
    d = pd.read_excel(path, sheet_name="Synthetic Dataset")
    d["si"] = (d.Impulse_Type == "Switching").astype(int)
    return d[d.Split == "Train"], d[d.Split != "Train"]


def cpri_knn(train, query):
    """CPRI's kNN exactly as their 'ML Helper' sheet computes it: squared
    distance with their per-feature scales (1500 kV, 1300/800/350 pF, 30 uH,
    efficiency unscaled, resistors /100 for LI and /30000 for SI, impulse
    type as a 0/1 mismatch), the 7 nearest, weights 1/(d + 1e-6), weighted
    mean of the residuals."""
    A = train[CPRI_X].to_numpy(float)
    R = train[CPRI_RS].to_numpy(float)
    base = np.array([1500, 1300, 800, 350, 30, 1.0])
    out = []
    for q in query[CPRI_X].to_numpy(float):
        r_scale = 100.0 if q[0] == 0 else 30000.0
        scale = np.r_[base, r_scale, r_scale]
        dist = (A[:, 0] != q[0]) + (((A[:, 1:] - q[1:]) / scale) ** 2).sum(1)
        idx = np.argsort(dist, kind="stable")[:7]
        w = 1.0 / (dist[idx] + 1e-6)
        out.append((R[idx] * w[:, None]).sum(0) / w.sum())
    return np.array(out)


def _cpri_score(test, pred, name):
    act, ph = test[CPRI_OB].to_numpy(), test[CPRI_PH].to_numpy()
    res = test[CPRI_RS].to_numpy()
    row = {"model": name}
    for j, k in enumerate(("front", "tail", "crest")):
        e = pred[:, j] - act[:, j]
        row[f"{k}_mae"] = float(np.abs(e).mean())
        row[f"{k}_rmse"] = float(np.sqrt((e ** 2).mean()))
        row[f"{k}_mape_%"] = float(100 * np.abs(e / act[:, j]).mean())
        # Share of the residual that the model explains. The observed-value
        # R2 CPRI reports (0.9999) is dominated by physics — physics alone
        # already gets 0.71% front-time error — so it cannot separate models.
        r = res[:, j] - (pred[:, j] - ph[:, j])
        row[f"{k}_resid_r2"] = float(1 - (r ** 2).sum() /
                                     ((res[:, j] - res[:, j].mean()) ** 2).sum())
    return row


def cpri_benchmark(path=CPRI_XLSX):
    """Every model corrects CPRI's physics columns using the same inputs their
    kNN sees. Relative residuals, one model per target, fixed
    hyperparameters (nothing tuned on the scored rows). Paired Wilcoxon of
    each model's per-row absolute error against CPRI's kNN."""
    from scipy.stats import wilcoxon
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    train, test = load_cpri(path)
    ph = test[CPRI_PH].to_numpy()
    preds = {"Physics only (no ML)": ph,
             "CPRI weighted kNN (their method)": ph + cpri_knn(train, test)}
    models = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
        "kNN k=7, standardised": lambda: make_pipeline(
            StandardScaler(), KNeighborsRegressor(7, weights="distance")),
        "Random forest": lambda: RandomForestRegressor(
            400, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "Gradient boosting": lambda: GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, random_state=0),
        "Neural network (MLP 64-64)": make_mlp,
    }
    Xtr, Xte = train[CPRI_X].to_numpy(float), test[CPRI_X].to_numpy(float)
    for name, mk in models.items():
        cols = []
        for j in range(3):
            y = train[CPRI_RS[j]].to_numpy() / train[CPRI_PH[j]].to_numpy()
            cols.append(mk().fit(Xtr, y).predict(Xte) * ph[:, j])
        preds[name] = ph + np.column_stack(cols)

    act = test[CPRI_OB].to_numpy()
    knn_err = np.abs(preds["CPRI weighted kNN (their method)"] - act)
    rows, paired = [], []
    for name, p in preds.items():
        rows.append(_cpri_score(test, p, name))
        if name.startswith("CPRI"):
            continue
        err = np.abs(p - act)
        for j, k in enumerate(("front", "tail", "crest")):
            pv = float(wilcoxon(err[:, j], knn_err[:, j]).pvalue)
            better = int((err[:, j] < knn_err[:, j]).sum())
            paired.append({"model": name, "target": k, "rows": len(test),
                           "better_than_knn": better,
                           "worse_than_knn": int((err[:, j] > knn_err[:, j]).sum()),
                           "wilcoxon_p": round(pv, 4),
                           "verdict": ("better" if pv < .05 and better > len(test) / 2
                                       else "worse" if pv < .05
                                       else "no significant difference")})
    return pd.DataFrame(rows), pd.DataFrame(paired), len(train), len(test)


def _cpri_checks(res):
    """The reimplementation must reproduce CPRI's reported numbers, or every
    comparison against it is a comparison against something else."""
    knn = res[res.model.str.startswith("CPRI")].iloc[0]
    ok = True
    for metric, key in (("mae", "mae"), ("rmse", "rmse"), ("mape", "mape_%")):
        for j, k in enumerate(("front", "tail", "crest")):
            got, want = knn[f"{k}_{key}"], CPRI_REPORTED[metric][j]
            p = abs(got - want) <= 1e-3 * max(1.0, want)
            ok &= p
            print(f"  [{'PASS' if p else 'FAIL'}] CPRI kNN {k:5s} {metric:4s} "
                  f"{got:9.4f}  (their sheet {want})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpri", action="store_true",
                    help="benchmark against CPRI's own dataset and kNN")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--severity", type=float, default=3.0)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--overshoot", action="store_true",
                    help="P8: evaluate overshoot prediction (lightning shots)")
    a = ap.parse_args()

    if a.cpri:
        res, paired, n_tr, n_te = cpri_benchmark()
        res.round(4).to_csv(os.path.join(RESULTS, "regression_cpri_results.csv"),
                            index=False)
        paired.to_csv(os.path.join(RESULTS, "regression_cpri_paired.csv"),
                      index=False)
        print(f"CPRI dataset: train {n_tr}, scored {n_te} (Validation + "
              f"Hidden Test, their protocol)\n")
        print("Reproduction of CPRI's reported kNN metrics")
        ok = _cpri_checks(res)
        pd.set_option("display.width", 200)
        print("\nMAPE % (front/peak, tail, crest)")
        print(res[["model", "front_mape_%", "tail_mape_%", "crest_mape_%"]]
              .round(3).to_string(index=False))
        print("\nShare of the residual explained (0 = no better than its mean)")
        print(res[["model", "front_resid_r2", "tail_resid_r2", "crest_resid_r2"]]
              .round(3).to_string(index=False))
        print("\nPaired against CPRI's kNN, per-row absolute error")
        print(paired.to_string(index=False))
        raise SystemExit(0 if ok else 1)

    if a.reuse and os.path.exists(DATA):
        df = pd.read_csv(DATA)
    else:
        df = build_dataset(a.n, severity=a.severity)
        df.to_csv(DATA, index=False)
        # Evaluate what was saved, not what is in memory: pandas' CSV round
        # trip drops the last bits of some floats, which flips tree splits,
        # so a fresh build and a later --reuse used to disagree slightly.
        df = pd.read_csv(DATA)
    print(f"{len(df)} shots  (LI {int((df.kind=='LI').sum())}, "
          f"SI {int((df.kind=='SI').sum())})")
    if a.build_only:
        raise SystemExit

    if a.overshoot:
        if "os" not in df.columns:
            raise SystemExit("this shot log has no overshoot columns; rebuild "
                             "it (python regression.py --n 360)")
        ov = overshoot_eval(df)
        ov.to_csv(os.path.join(RESULTS, "regression_overshoot.csv"), index=False)
        print(ov.to_string(index=False))
        raise SystemExit
    if a.curve:
        lc = learning_curve(df)
        print(lc.pivot(index="train_shots", columns="model",
                       values="t1_mape_%").round(2).to_string())
        print(lc.pivot(index="train_shots", columns="model",
                       values="compliance_acc_%").round(1).to_string())
        raise SystemExit
    res = evaluate_all(df)
    res.to_csv(os.path.join(RESULTS, "regression_results.csv"), index=False)
    cols = ["split", "model", "t1_mape_%", "t2_mape_%", "eff_mape_%",
            "compliance_acc_%"]
    print(res[cols].to_string(index=False))
