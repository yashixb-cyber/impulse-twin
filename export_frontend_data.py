"""
export_frontend_data.py — precompute everything the browser frontend needs,
so the page does real physics-derived numbers with no server and no JS port
of circuit.py/waveform.py (an independent reimplementation is exactly the
"wrong circuit model" risk CLAUDE.md rule 1 warns about).

For a fixed (kind, test-object capacitance) the circuit is linear, so waveform
SHAPE (and so which stage/Rf/Rt combo is best) does not depend on charging
voltage — design.py already relies on this. So we precompute the ranked
shape table once per point on a capacitance grid, store each candidate's
"per_volt" response, and let the browser do the last step (target voltage ->
charging voltage -> feasibility -> tie-break) with plain arithmetic. That
last step is linear scaling and a stock lookup, not a physics simulation, so
doing it in JS carries none of the reimplementation risk.

    python export_frontend_data.py              # writes frontend_data.json (~8 min)
    python export_frontend_data.py --reuse-grid # keep the physics grid and loop
                                                # tables, redo everything learned
                                                # from the shot log (~2 min)
"""
import json
import os
import time

import numpy as np
import pandas as pd

APPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
FRONTEND = os.path.join(APPDIR, "frontend_data.json")

import atmosphere as A
import regression as R
import strategies as S
from circuit import Generator, Setting, resistor_options
from design import Belief, candidates, predict, predict_fast
from recommend import EQUIPMENT, IEC_60071, TYPICAL_UM

G = Generator()


def _setting_frame(kind, c_pf, rows, v_kv):
    """Grid rows as the columns regression.py's models read. Textbook
    features come from the row itself; waveform shape is independent of
    charging voltage (linear circuit), so unit-charge values hold at any."""
    return pd.DataFrame([{
        "kind": kind, "is_si": int(kind == "SI"), "n": r["n"],
        "v_charge_kv": v_kv, "rf_stage": r["rf"], "rt_stage": r["rt"],
        "c_load_nf": r["cl"] * 1e9, "c_test_pf": c_pf,
        "th_t1": r["t1"], "th_t2": r["t2"], "th_eff": r["pv"] / r["n"],
        "v0_kv": r["n"] * v_kv} for r in rows])


def hybrid_block(grid, n_vectors=60, seed=0):
    """The hybrid regression.py validates, fitted on the whole shot log and
    made runnable in the browser.

    Physics calibration needs the circuit solver, so its prediction is
    precomputed here for every grid row, like the rest of the grid. Only the
    boosted trees run in the browser — their input includes the charging
    voltage, which is not known until the operator picks a target. Returns
    (block for the app, test vectors for the browser-code check)."""
    if not os.path.exists(R.DATA):
        print("no regression_dataset.csv — the app ships physics-only")
        return None, None
    log = pd.read_csv(R.DATA)
    b, gb = R.fit_hybrid(log)

    for kind in ("LI", "SI"):
        for pt in grid[kind]:
            if not pt["rows"]:
                continue
            ph = R.physics_predict(b, _setting_frame(kind, pt["c_pf"], pt["rows"], 1e-3),
                                   overshoot=kind == "LI")
            for r, p in zip(pt["rows"], ph):
                r["ph"] = [float(v) for v in p]

    models = [R.gb_to_json(e) for e in gb.estimators_]
    # The serialised trees must reproduce sklearn on every logged shot.
    X = np.column_stack([log[R.FEATURES].to_numpy(), R.physics_predict(b, log)])
    for m, est in zip(models, gb.estimators_):
        err = np.abs(R.gb_json_predict(m, X) - est.predict(X)).max()
        assert err < 1e-12, f"tree export drifted from sklearn by {err}"
    osm = R.fit_overshoot(log, b) if "os" in log.columns else None
    if osm is not None:
        li = log[log.kind == "LI"]
        Xo = np.column_stack([li[R.FEATURES].to_numpy(), R.physics_predict(b, li, overshoot=True)])
        os_model = R.gb_to_json(osm)
        err = np.abs(R.gb_json_predict(os_model, Xo) - osm.predict(Xo)).max()
        assert err < 1e-12, f"overshoot tree export drifted from sklearn by {err}"

    # Test vectors: random grid rows at random charging voltages, expected
    # values from regression.hybrid_predict itself. That recomputes the
    # calibrated physics independently of the "ph" stored above, so the
    # browser check covers feature order, units and the precomputation.
    rng = np.random.default_rng(seed)
    picks = [(k, i, j) for k in ("LI", "SI") for i, pt in enumerate(grid[k])
             for j in range(len(pt["rows"]))]
    vectors = []
    for n in rng.choice(len(picks), n_vectors, replace=False):
        kind, i, j = picks[n]
        pt = grid[kind][i]
        v_kv = float(rng.uniform(40, 180))
        _, hy = R.hybrid_predict(b, gb, _setting_frame(kind, pt["c_pf"], [pt["rows"][j]], v_kv))
        vectors.append({"kind": kind, "point": i, "row": j, "v_charge_kv": v_kv,
                        "expected": [float(v) for v in hy[0]]})
    os_vectors = []
    if osm is not None:
        li_picks = [(i, j) for i, pt in enumerate(grid["LI"]) for j in range(len(pt["rows"]))]
        for n in rng.choice(len(li_picks), 30, replace=False):
            i, j = li_picks[n]
            pt, v_kv = grid["LI"][i], float(rng.uniform(40, 180))
            _, hy = R.overshoot_predict(b, osm, _setting_frame("LI", pt["c_pf"], [pt["rows"][j]], v_kv))
            os_vectors.append({"point": i, "row": j, "v_charge_kv": v_kv, "expected": float(hy[0])})

    res_path = os.path.join(R.RESULTS, "regression_results.csv")
    cv = pd.read_csv(res_path) if os.path.exists(res_path) else None
    pick = lambda m: float(cv[(cv.split == "random CV") & (cv.model == m)]["t1_mape_%"].iloc[0])

    block = {
        "features": R.HYBRID_FEATURES, "targets": R.TARGET_COLS, "models": models,
        "shots": int(len(log)),
        "c_train_pf": [float(log.c_test_pf.min()), float(log.c_test_pf.max())],
        "v_train_kv": [float(log.v_charge_kv.min()), float(log.v_charge_kv.max())],
        "os_model": R.gb_to_json(osm) if osm is not None else None,
        "ringing": R.RINGING,
        "cv_t1_mape": {"hybrid": pick("Hybrid: physics + boosting (ours)"),
                       "knn": pick("Weighted KNN (CPRI baseline)")} if cv is not None else None,
        "belief": {"l_factor": b.l_factor, "l_extra_uh": b.l_extra * 1e6,
                   "c2_offset_pf": b.c2_offset * 1e12, "tm_ns": b.tm * 1e9,
                   "eff_scale": b.eff_scale},
        "provenance": (f"Learned from {len(log)} shots of a SIMULATED impulse hall "
                       f"(regression.py). A stand-in for CPRI's own shot log: retrain "
                       f"on real reduced-voltage trial shots before relying on it."),
    }
    return block, {"hybrid": vectors, "os": os_vectors}


def log_block():
    """The shot log itself, for the page's similar-shots evidence: FEATURES,
    then measured t1, t2, eff, overshoot."""
    log = pd.read_csv(R.DATA)
    cols = R.FEATURES + ["t1", "t2", "eff", "os"]
    return {"features": R.FEATURES, "cols": cols,
            "rows": [[float(f"{v:.6g}") for v in r] for r in log[cols].to_numpy()]}


def similar_vectors(grid, log_json, n=24, seed=1):
    """Expected neighbours and KNN predictions from regression.similar_shots,
    for grid rows queried the way the page queries them. The log is rebuilt
    from the rounded rows the page receives, so both sides see one table."""
    log = pd.DataFrame(log_json["rows"], columns=log_json["cols"])
    log["kind"] = np.where(log.is_si == 1, "SI", "LI")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kind = ("LI", "SI")[int(rng.integers(2))]
        pt = grid[kind][int(rng.integers(len(grid[kind])))]
        row = pt["rows"][int(rng.integers(len(pt["rows"])))]
        q = _setting_frame(kind, pt["c_pf"], [row], float(rng.uniform(40, 180)))
        idx, pred = R.similar_shots(log, q)
        out.append({"kind": kind, "query": [float(v) for v in q[R.FEATURES].iloc[0]],
                    "idx": [int(i) for i in idx], "pred": [float(v) for v in pred]})
    return out


def memory_vectors(loop, seed=2):
    """strategies.AppMemory against the page's memoryCorrection, and
    strategies.app_memory's first shot (design.py with the correction)
    against the page's memoryFirstShot on the same loop-table point."""
    from design import Correction, design
    rng = np.random.default_rng(seed)
    cats = ["porcelain_disc_string", "composite_long_rod", "post_insulator_stack"]
    corr_cases = []
    for _ in range(20):
        mem = S.AppMemory()
        for _ in range(int(rng.integers(0, 9))):
            mem.tests.append({"kind": ("LI", "SI")[int(rng.integers(2))],
                              "category": cats[int(rng.integers(3))],
                              "t1": float(rng.uniform(.85, 1.2)), "t2": float(rng.uniform(.9, 1.1)),
                              "os": float(rng.uniform(-.01, .09)), "gain": float(rng.uniform(.9, 1.1))})
        kind, cat = ("LI", "SI")[int(rng.integers(2))], cats[int(rng.integers(3))]
        c = mem.correction(kind, cat)
        corr_cases.append({"tests": mem.tests, "kind": kind, "category": cat,
                           "expected": None if c is None else
                           {"t1": c[0].t1, "t2": c[0].t2, "os": c[0].overshoot, "gain": c[1]}})
    first = []
    for _ in range(20):
        kind = ("LI", "SI")[int(rng.integers(2))]
        i = int(rng.integers(len(loop[kind])))
        target = float(rng.choice([1300e3, 1425e3, 1550e3] if kind == "LI" else [950e3, 1050e3]))
        corr = {"t1": float(rng.uniform(.88, 1.15)), "t2": float(rng.uniform(.92, 1.08)),
                "os": float(rng.uniform(0, .06)) if kind == "LI" else 0.0,
                "gain": float(rng.uniform(.93, 1.08))}
        cand = design(kind, target, loop[kind][i]["c_pf"] * 1e-12,
                      Belief(eff_scale=1 / corr["gain"]), G, top=1,
                      corr=Correction(corr["t1"], corr["t2"], corr["os"]))
        if not cand:
            continue
        st = cand[0]["setting"]
        first.append({"kind": kind, "point": i, "target": target, "corr": corr,
                      "expected": [st.n, st.rf_stage, st.rt_stage, st.c_load]})
    return corr_cases, first


def demo_memory(n_tests=10, hall_seed=4242, severity=3.0, seed=7):
    """Tests from the SIMULATED hall the shot log comes from, run through
    strategies.app_memory online, in the page's record format — so the page's
    memory can be shown working before a lab has stored any tests of its own.
    Labelled as demo data wherever the page shows it."""
    from lab import hall, make_session
    lab, rng, mem, out = hall(hall_seed, severity), np.random.default_rng(seed), S.AppMemory(), []
    for k in range(n_tests):
        obj = make_session(rng, kind="LI" if k % 3 else "SI")
        lab.reseed_noise(seed, k)
        shots = S.app_memory(lab, obj, 70_000 + k, mem, G)
        mem.add(obj, shots, G)
        si, c_pf = int(obj.kind == "SI"), obj.c_nominal * 1e12
        rows = []
        for x in shots:
            st, r = x["setting"], x["result"]
            th = predict(st, G, obj.c_nominal, Belief(), obj.kind)[0]
            v0 = st.n * st.v_charge
            rows.append([si, st.n, st.v_charge / 1e3, st.rf_stage, st.rt_stage, st.c_load * 1e9,
                         c_pf, th.t1, th.t2, th.test_voltage / v0,
                         r.t1, r.t2, r.test_voltage / v0, r.overshoot])
        out.append({**mem.tests[-1], "c_pf": c_pf, "source": "demo",
                    "shots": [[float(f"{v:.6g}") for v in row] for row in rows]})
    return out


def atmo_vectors(n=24, seed=3):
    """Random site conditions, each combination exercised the three ways an
    operator can give them (nothing, humidity given, pressure vs altitude),
    against atmosphere.correction_factor — the app's atmosphericCorrection()
    must match it exactly (k1 is exact on both sides; k2 uses the same
    declared approximation on both sides, so this is a real equality, not
    a tolerance)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        temp = float(rng.uniform(-10, 50))
        give_rh = bool(rng.integers(2))
        rh = float(rng.uniform(5, 100)) if give_rh else None
        by_alt = bool(rng.integers(2))
        alt = float(rng.uniform(0, 3000)) if by_alt else None
        press = None if by_alt else float(rng.uniform(80, 103))
        kw = {k: v for k, v in dict(temp_c=temp, rh_pct=rh, pressure_kpa=press,
                                    altitude_m=alt).items() if v is not None}
        r = A.correction_factor(**kw)
        out.append({"tempC": temp, "rhPct": rh, "pressureKpa": press, "altitudeM": alt,
                    "expected": {"kt": r["kt"], "k1": r["k1"], "k2": r["k2"],
                                "k1Reliable": r["k1_reliable"]}})
    return out


def _loop_point(args):
    """Every stocked setting at one capacitance, with its textbook
    prediction, as compact arrays [n, rf, rt, cl, t1, t2, os, pv]."""
    kind, c_pf = args
    rows = []
    for n, rf, rt, cl in candidates(kind, G):
        s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl)
        r, _, _ = predict(s, G, c_pf * 1e-12, Belief(), kind)
        if r.test_voltage > 0 and np.isfinite(r.t1) and np.isfinite(r.t2):
            rows.append([n, rf, rt, cl, round(r.t1, 4), round(r.t2, 3),
                         round(r.overshoot, 4), float(f"{r.test_voltage:.6g}")])
    return {"c_pf": round(float(c_pf), 2), "rows": rows}


def loop_tables(n_points=12):
    """What the app's shot loop searches after a shot: the WHOLE stock, not
    the recommendation shortlist. After a measured shot the right setting
    is often one the textbook ranked badly — a front measured at 64% of
    prediction calls for 60 ohm where the textbook preferred 30 — and the
    shortlist does not contain it. strategies.app_loop, which the benchmark
    validates, searches the whole stock, so the browser must too. A coarser
    grid than the shortlist's: each shot's measured ratios reset the level,
    and differences between candidates vary slowly with capacitance."""
    from concurrent.futures import ProcessPoolExecutor
    grid_pf = np.geomspace(20, 1200, n_points)
    jobs = [(k, float(c)) for k in ("LI", "SI") for c in grid_pf]
    with ProcessPoolExecutor() as pool:
        pts = list(pool.map(_loop_point, jobs))
    return {k: [p for (kk, _), p in zip(jobs, pts) if kk == k] for k in ("LI", "SI")}


COMPOSITION_P = (2, 3, 4, 6)     # whole-bank p parallel columns x (12 / p) series
COMPOSITION_KEEP = 10            # best rows per connection per capacitance point


def _composition_point(args):
    """Series-parallel connections at one capacitance, at unit charge: the same
    columns as the grid rows plus p. Physics only: the learned correction was
    trained on all-in-series shots, so it is not applied to these."""
    kind, c_pf = args
    rows = []
    for p in COMPOSITION_P:
        part = []
        for n, rf, rt, cl in candidates(kind, G, p):
            s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl, p=p)
            r, _, _ = predict(s, G, c_pf * 1e-12, Belief(), kind)
            if r.test_voltage <= 0 or not (np.isfinite(r.t1) and np.isfinite(r.t2)):
                continue
            part.append({"n": n, "p": p, "rf": rf, "rt": rt, "cl": cl,
                         "t1": round(r.t1, 4), "t2": round(r.t2, 3),
                         "os": round(r.overshoot, 4),
                         "margin": round(r.margin(1.0, shape_only=True), 4),
                         "pv": r.test_voltage})
        part.sort(key=lambda d: d["margin"])
        rows += part[:COMPOSITION_KEEP]
    return {"c_pf": round(float(c_pf), 2), "rows": rows}


INDUCTANCE_UH = (15, 30, 60, 120)    # external loop inductance levels; nominal 5 is the main grid


def _inductance_point(args):
    kind, c_pf, l_uh = args
    b = Belief(l_extra=l_uh * 1e-6 - G.l_ext)
    return shape_table(kind, c_pf * 1e-12, belief=b)


def inductance_block(n_points=40):
    """The same shape table as the main grid at other external loop
    inductances. The app interpolates between the levels by setting."""
    from concurrent.futures import ProcessPoolExecutor
    grid_pf = np.geomspace(20, 1200, n_points)
    jobs = [(k, float(c), l) for l in INDUCTANCE_UH for k in ("LI", "SI") for c in grid_pf]
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(_inductance_point, jobs, chunksize=4))
    out = {"levels": [G.l_ext * 1e6] + list(INDUCTANCE_UH), "LI": {}, "SI": {}}
    for (k, c, l), rows in zip(jobs, res):
        out[k].setdefault(str(l), []).append({"c_pf": round(c, 2), "rows": rows})
    return out


def cpri_stock_block():
    """CPRI's own calculator stock and the settings in its dataset, for the
    app's "is it buildable from CPRI's equipment" check. Read from the sheet,
    not typed: data/cpri_optimiser.xlsx."""
    import pandas as pd
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cpri_optimiser.xlsx")
    inv = pd.read_excel(path, sheet_name="Resistor Inventory", header=2)
    units = int(inv.iloc[:, 1].dropna().iloc[0])
    ds = pd.read_excel(path, sheet_name="Synthetic Dataset")
    out = {"front": [float(x) for x in inv.iloc[:, 0].dropna()],
           "tail": [float(x) for x in inv.iloc[:, 4].dropna()],
           "units": units, "dataset": {}}
    for kind, name in (("LI", "Lightning"), ("SI", "Switching")):
        g = ds[ds.Impulse_Type == name]
        out["dataset"][kind] = {
            "rows": int(len(g)),
            "rf": [round(float(g.Front_R_Stage.min()), 1), round(float(g.Front_R_Stage.max()), 1)],
            "rt": [round(float(g.Tail_R_Stage.min()), 1), round(float(g.Tail_R_Stage.max()), 1)],
            "stages": [int(g.Stages.min()), int(g.Stages.max())],
            "charge_kv": [round(float(g.Charge_kV_Stage.min()), 1), round(float(g.Charge_kV_Stage.max()), 1)]}
    return out


def composition_block(n_points=40):
    from concurrent.futures import ProcessPoolExecutor
    grid_pf = np.geomspace(20, 1200, n_points)
    jobs = [(k, float(c)) for k in ("LI", "SI") for c in grid_pf]
    with ProcessPoolExecutor() as pool:
        pts = list(pool.map(_composition_point, jobs))
    return {k: [p for (kk, _), p in zip(jobs, pts) if kk == k] for k in ("LI", "SI")}


def add_compositions():
    """--compositions-only: add the series-parallel table to an existing
    frontend_data.json without rebuilding the grid or the learned parts."""
    t0 = time.time()
    with open(FRONTEND, encoding="utf-8") as f:
        data = json.load(f)
    data["compositions"] = composition_block()
    data["generator"]["compositions"] = list(COMPOSITION_P)
    data["cpri_stock"] = cpri_stock_block()
    if "--inductance" in sys.argv:
        t1 = time.time()
        data["inductance"] = inductance_block()
        print(f"inductance levels added, {time.time() - t1:.0f}s")
    with open(FRONTEND, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"compositions added ({sum(len(p['rows']) for k in data['compositions'] for p in data['compositions'][k])} rows), "
          f"{time.time() - t0:.0f}s")


def shape_table(kind, c_test, keep=40, belief=None):
    """Ranked, target-voltage-independent shortlist: same screen-then-exact
    procedure as design.design(), minus the per-target_v feasibility filter."""
    pool = list(candidates(kind, G))
    belief = belief or Belief()
    if kind == "LI" and len(pool) > 80:
        scored = []
        for n, rf, rt, cl in pool:
            s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl)
            r, _, _ = predict_fast(s, G, c_test, belief, kind)
            scored.append((r.margin(1.0, shape_only=True), (n, rf, rt, cl)))
        scored.sort(key=lambda x: x[0])
        pool = [c for _, c in scored[:60]]

    rows = []
    for n, rf, rt, cl in pool:
        s = Setting(n=n, v_charge=1.0, rf_stage=rf, rt_stage=rt, c_load=cl)
        r, _, _ = predict(s, G, c_test, belief, kind)
        if r.test_voltage <= 0:
            continue
        rows.append({
            "n": n, "rf": rf, "rt": rt, "cl": cl,
            "t1": round(r.t1, 4), "t2": round(r.t2, 3),
            "os": round(r.overshoot, 4),
            "margin": round(r.margin(1.0, shape_only=True), 4),
            "pv": r.test_voltage,
        })
    rows.sort(key=lambda d: d["margin"])
    return rows[:keep]


def build(reuse_grid=False):
    if reuse_grid:
        return rebuild_learned()
    # Log-spaced grid covering every equipment type's measured-capacitance
    # allowance (0.3x-3x its nominal range) and circuit.py's validated band
    # (30-1100 pF). 40 points -> ~11% steps, so "nearest grid point" is
    # never more than ~5% off in capacitance.
    grid_pf = np.geomspace(20, 1200, 40)

    out = {"LI": [], "SI": []}
    t0 = time.time()
    for kind in ("LI", "SI"):
        for c_pf in grid_pf:
            rows = shape_table(kind, float(c_pf) * 1e-12)
            out[kind].append({"c_pf": round(float(c_pf), 2), "rows": rows})
        print(f"{kind} grid done, {time.time() - t0:.0f}s elapsed")

    meta = {
        "n_stages": G.n_stages, "v_stage_max": G.v_stage_max,
        "charge_margin": G.charge_margin, "c_stage": G.c_stage,
        "rf_li": list(G.rf_li), "rt_li": list(G.rt_li),
        "rf_si": list(G.rf_si), "rt_si": list(G.rt_si),
        "cload_si": list(G.cload_si),
        "c_divider": G.c_divider, "c_stray": G.c_stray_nominal,
        "c_base_load": G.c_base_load,
        # How to physically build each per-stage value from the nameplate
        # units. Keys use :g so they match JS String(value) ("232", not
        # "232.0"); the app looks each recommended Rf/Rt up by that key.
        "build": {kind: {
            "rf": {f"{v:g}": d for v, d in resistor_options(G.front_units).items()},
            "rt": {f"{v:g}": d for v, d in resistor_options(units).items()},
        } for kind, units in (("LI", G.tail_li_units), ("SI", G.tail_si_units))},
    }
    equipment = {k: {"range_pf": v[0], "label": v[1]} for k, v in EQUIPMENT.items()}
    iec = {str(k): {"LI": list(v[0]), "SI": list(v[1])} for k, v in IEC_60071.items()}
    typical_um = {k: list(v) for k, v in TYPICAL_UM.items()}

    loop = loop_tables()
    comps = composition_block()
    meta["compositions"] = list(COMPOSITION_P)
    print(f"shot-loop tables ({sum(len(p['rows']) for k in loop for p in loop[k])} "
          f"settings), {time.time() - t0:.0f}s elapsed")
    data = {"grid": out, "generator": meta, "equipment": equipment,
            "iec60071": iec, "typical_um": typical_um, "loop": loop,
            "compositions": comps,
            "cpri_stock": cpri_stock_block(),
            "inductance": inductance_block()}
    _attach_learned(data, t0)


def rebuild_learned():
    """--reuse-grid: the physics grid and loop tables depend only on the
    generator and stock; everything else here comes from the shot log."""
    with open(FRONTEND, encoding="utf-8") as f:
        data = json.load(f)
    _attach_learned(data, time.time())


def _attach_learned(data, t0):
    hybrid, vectors = hybrid_block(data["grid"])
    data["hybrid"] = hybrid
    print(f"hybrid attached, {time.time() - t0:.0f}s elapsed")
    data["log"] = log_block() if hybrid else None
    app_vec = {"atmo": atmo_vectors()}
    if hybrid:
        mem_corr, mem_first = memory_vectors(data["loop"])
        app_vec.update(os=vectors["os"], similar=similar_vectors(data["grid"], data["log"]),
                       mem_corr=mem_corr, mem_first=mem_first)
        print(f"test vectors, {time.time() - t0:.0f}s elapsed")
    data["demo_memory"] = demo_memory()
    print(f"demo memory ({len(data['demo_memory'])} tests), {time.time() - t0:.0f}s elapsed")
    with open(FRONTEND, "w", encoding="utf-8") as f:
        json.dump(data, f)
    if hybrid:
        with open(os.path.join(APPDIR, "hybrid_test_vectors.json"), "w", encoding="utf-8") as f:
            json.dump(vectors["hybrid"], f)
    with open(os.path.join(APPDIR, "app_test_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(app_vec, f)
    print(f"frontend_data.json written, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    import sys
    if "--compositions-only" in sys.argv:
        add_compositions()
    else:
        build(reuse_grid="--reuse-grid" in sys.argv)
