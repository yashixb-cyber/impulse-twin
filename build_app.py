"""
build_app.py — assembles the single-file operator + evidence app.

Two data sources, both already-authoritative Python output, embedded as JSON
with no server needed:

  * frontend_data.json  — the recommend-settings grid (export_frontend_data.py)
  * dashboard.py's own data assembly (session, benchmark, regression) —
    reused via import, not reimplemented, so this file never drifts from
    what `python run_all.py` verifies.

    python export_frontend_data.py     # once, ~4 min, rebuild if the stock
                                        # or generator spec in circuit.py changes
    python build_app.py                # writes impulse_app.html; minutes, it
                                        # repeats dashboard.py's session search
    python build_app.py --reuse-session  # seconds: keep the existing session
"""
import json
import os

import dashboard as D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")


def dashboard_data():
    """Same session block dashboard.py builds for its own page — reused so
    the evidence tab can never disagree with `python dashboard.py`. The
    session is a replayed benchmark session (dashboard.pick_from_benchmark)."""
    return {**D.session_data(*D.pick_from_benchmark()), **results_data()}


def results_data():
    """The benchmark and regression tables, read fresh from their CSVs."""
    summ_path = os.path.join(RESULTS, "benchmark_summary.csv")
    meta_path = os.path.join(RESULTS, "benchmark_meta.json")
    rpath = os.path.join(RESULTS, "regression_results.csv")
    lpath = os.path.join(RESULTS, "regression_learning_curve.csv")
    read = lambda p: D.pd.read_csv(p).to_dict("records") if os.path.exists(p) else []
    at = lambda f: os.path.join(RESULTS, f)
    return {"bench": read(summ_path), "reg": read(rpath), "lc": read(lpath),
            "by_kind": read(at("benchmark_by_kind.csv")),
            "paired": read(at("benchmark_paired.csv")),
            "paired_kind": read(at("benchmark_paired_by_kind.csv")),
            "os_reg": read(at("regression_overshoot.csv")),
            # CPRI's own dataset (regression.py --cpri): MAPE and the
            # residual-R2 the R2 they report can't show (see CLAUDE.md §5),
            # plus the row-by-row paired test against their kNN.
            "cpri_reg": read(at("regression_cpri_results.csv")),
            "cpri_paired": read(at("regression_cpri_paired.csv")),
            # tune_cpri.py: every model, CPRI's kNN included, tuned on Train only.
            "cpri_tuned": read(at("regression_cpri_tuned.csv")),
            "cpri_tuned_paired": read(at("regression_cpri_tuned_paired.csv")),
            "meta": json.load(open(meta_path, encoding="utf-8"))
            if os.path.exists(meta_path) else {}}


def reused_dashboard_data(path):
    """--reuse-session: the illustrative session from the existing app, with
    every results table refreshed from its CSV. Replaying the benchmark
    session is the only slow part of a build (a few minutes) and it only
    changes when strategies.py, lab.py or the benchmark do — then build
    without the flag."""
    old = open(path, encoding="utf-8").read()
    i = old.index("DASH = ") + len("DASH = ")
    dash, _ = json.JSONDecoder().raw_decode(old[i:])
    dash.update(results_data())
    return dash


_NODE_CHECK = r"""
const fs = require('fs');
const [dataPath, vecPath, targetsJson] = process.argv.slice(2);
const D = JSON.parse(fs.readFileSync(dataPath, 'utf8')), H = D.hybrid;
const V = JSON.parse(fs.readFileSync(vecPath, 'utf8')), T = JSON.parse(targetsJson);
let rel = 0;
for(const v of V){
  const pt = D.grid[v.kind][v.point], row = pt.rows[v.row];
  const got = hybridAt(H, row, v.kind, pt.c_pf, v.v_charge_kv);
  got.forEach((g, k) => rel = Math.max(rel, Math.abs(g / v.expected[k] - 1)));
}
let marg = 0, solved = 0, fixpt = 0;
for(const kind of ['LI', 'SI']) for(const pt of D.grid[kind]) for(const row of pt.rows){
  marg = Math.max(marg, Math.abs(shapeMargin(T[kind], row.t1, row.t2, row.os) - row.margin));
  const s = hybridSolve(H, row, kind, pt.c_pf, 1.0e6);
  if(s){ solved++; fixpt = Math.max(fixpt, Math.abs(s.eff*row.n*s.v_charge/1.0e6 - 1)); }
}
// Shot loop: the rules of strategies.app_loop, on a real full-stock table.
const TL = T.LI, gen = D.generator, target = 1.425e6;
const lp = D.loop.LI.find(p => p.c_pf > 80);
const pt = {c_pf: lp.c_pf, rows: lp.rows.map(([n, rf, rt, cl, t1, t2, os, pv]) =>
                                    ({n, rf, rt, cl, t1, t2, os, pv}))};
// fire the best-centred feasible 12-stage setting, as the app would
const fired = pt.rows.filter(r => r.n === 12 && target / r.pv <= gen.charge_margin * gen.v_stage_max)
                     .sort((a, b) => shapeMargin(TL, a.t1, a.t2, a.os) - shapeMargin(TL, b.t1, b.t2, b.os))[0];
const vch = target / fired.pv, loop = {};
const shot = meas => [{row: fired, v_charge: vch, meas}];
let n = nextShot(TL, pt.rows, shot({t1: fired.t1, t2: fired.t2, v: 0.96*target, os: fired.os}), target, gen);
loop.voltage_only = n.voltageOnly === true && n.row === fired && n.climbs === 0
                    && Math.abs(n.v_charge / (vch / 0.96) - 1) < 1e-12;
loop.done = nextShot(TL, pt.rows, shot({t1: fired.t1, t2: fired.t2, v: target, os: fired.os}), target, gen).done === true;
n = nextShot(TL, pt.rows, shot({t1: 0.5*TL.t1, t2: fired.t2, v: target, os: 0}), target, gen);
loop.shape_fail_moves = !!n.row && rowKey(n.row) !== rowKey(fired) && n.row.n === fired.n
                        && n.margin <= COMFORTABLE && n.climbs >= 1;
// measured test voltage 10% under the prediction -> next charge scaled by 1/0.9
n = nextShot(TL, pt.rows, shot({t1: 0.5*TL.t1, t2: fired.t2, v: 0.9*target, os: 0}), target, gen);
const cap = gen.charge_margin * gen.v_stage_max, want = Math.min(target / n.row.pv / 0.9, cap);
loop.gain_applied = Math.abs(n.v_charge / want - 1) < 1e-12;
// App vectors: overshoot model, similar shots, hall memory.
const A = JSON.parse(fs.readFileSync(process.argv[5], 'utf8')), app = {};
app.os = Math.max(...A.os.map(v => { const pt = D.grid.LI[v.point], row = pt.rows[v.row];
  return Math.abs(osPredict(H, row, pt.c_pf, v.v_charge_kv) - v.expected); }));
let simIdx = 0, simRel = 0;
for(const v of A.similar){
  const s = similarShots(D.log, v.kind, v.query, 5);
  simIdx += s.idx.join() === v.idx.join() ? 1 : 0;
  s.pred.forEach((p, k) => simRel = Math.max(simRel, Math.abs(p / v.pred[k] - 1)));
}
app.similar_idx = simIdx + '/' + A.similar.length; app.similar_rel = simRel;
app.mem_corr = Math.max(...A.mem_corr.map(v => { const c = memoryCorrection(v.tests, v.kind, v.category);
  if(!c || !v.expected) return (c === null) === (v.expected === null) ? 0 : 1;
  return Math.max(...['t1', 't2', 'os', 'gain'].map(k => Math.abs(c[k] - v.expected[k]))); }));
let agree = 0;
for(const v of A.mem_first){
  const lp = D.loop[v.kind][v.point], rows = lp.rows.map(([n, rf, rt, cl, t1, t2, os, pv]) =>
    ({n, rf, rt, cl, t1, t2, os, pv}));
  const f = memoryFirstShot(T[v.kind], rows, v.corr, v.target, gen);
  agree += f && [f.row.n, f.row.rf, f.row.rt, f.row.cl].join() === v.expected.join() ? 1 : 0;
}
app.mem_first_agree = agree + '/' + A.mem_first.length;
app.atmo = Math.max(...A.atmo.map(v => {
  const r = atmosphericCorrection(v.tempC, v.rhPct, v.pressureKpa, v.altitudeM);
  return Math.max(Math.abs(r.kt - v.expected.kt), Math.abs(r.k1 - v.expected.k1),
                  Math.abs(r.k2 - v.expected.k2), r.k1Reliable === v.expected.k1Reliable ? 0 : 1);
}));
console.log(JSON.stringify({vectors: V.length, rel, marg, solved, fixpt, loop, app}));
"""


def check_hybrid_js():
    """Run the page's own HYBRID block in Node against regression.py.

    Test vectors come from regression.hybrid_predict with the calibrated
    physics recomputed from scratch, so a wrong feature order, unit or
    precomputed value fails here — not in front of the jury. The margin
    formula is checked against the Python margin of every grid row."""
    import shutil
    import subprocess
    import tempfile
    from waveform import TARGETS
    vec = os.path.join(APPDIR, "hybrid_test_vectors.json")
    avec = os.path.join(APPDIR, "app_test_vectors.json")
    node = shutil.which("node")
    if not node or not os.path.exists(vec) or not os.path.exists(avec):
        print("hybrid JS check SKIPPED — "
              + ("node not found" if not node else "no test vectors; run "
                 "export_frontend_data.py"))
        return
    block = TEMPLATE.split("/* HYBRID-BEGIN")[1].split("/* HYBRID-END */")[0]
    block = block.split("*/", 1)[1]           # drop the rest of the opening comment
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(block + _NODE_CHECK)
    try:
        r = subprocess.run([node, f.name, os.path.join(APPDIR, "frontend_data.json"),
                            vec, json.dumps(TARGETS), avec],
                           capture_output=True, text=True, check=True)
    finally:
        os.unlink(f.name)
    res = json.loads(r.stdout)
    # 1e-6, not 1e-12: the browser uses physics precomputed at unit charge,
    # the vectors recompute it at the real charging voltage, and the IEC
    # base-curve fit converges to ~1e-9 (7e-10 observed). A wrong feature
    # order or unit shows up at ~1e-2, so this threshold still catches it.
    ok = res["rel"] < 1e-6 and res["marg"] < 2e-3 and res["fixpt"] < 1e-3
    loop_ok = all(res["loop"].values())
    print(f"hybrid JS check {'PASSED' if ok else 'FAILED'}: {res['vectors']} vectors, "
          f"max relative error vs regression.py {res['rel']:.1e}; margin formula "
          f"max diff {res['marg']:.1e}; {res['solved']} rows solved, charge "
          f"fixed-point residual {res['fixpt']:.1e}")
    print(f"shot-loop JS check {'PASSED' if loop_ok else 'FAILED'}: "
          + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in res["loop"].items()))
    a = res["app"]
    n_first = int(a["mem_first_agree"].split("/")[1])
    # Overshoot and memory arithmetic must match exactly (to float noise);
    # neighbours must be the same shots. The memory first shot may differ
    # from strategies.app_memory when design.py's 60-setting LI shortlist
    # excludes the setting the app, searching the whole stock, prefers —
    # so that is reported and only required to agree in 90% of cases.
    app_ok = (a["os"] < 1e-9 and a["similar_idx"].split("/")[0] == a["similar_idx"].split("/")[1]
              and a["similar_rel"] < 1e-9 and a["mem_corr"] < 1e-12
              and int(a["mem_first_agree"].split("/")[0]) >= 0.9 * n_first
              and a["atmo"] < 1e-9)
    print(f"app JS check {'PASSED' if app_ok else 'FAILED'}: overshoot max diff "
          f"{a['os']:.1e}; similar shots same neighbours {a['similar_idx']}, KNN "
          f"prediction max relative diff {a['similar_rel']:.1e}; memory correction "
          f"max diff {a['mem_corr']:.1e}; memory first shot agrees with "
          f"strategies.app_memory in {a['mem_first_agree']}; atmospheric correction "
          f"max diff vs atmosphere.py {a['atmo']:.1e}")
    ok = ok and loop_ok and app_ok
    if not ok:
        raise SystemExit("the browser's hybrid does not match regression.py — "
                         "impulse_app.html not written")


SUITES = [("circuit", ["circuit.py"]), ("waveform", ["waveform.py"]),
          ("recommend", ["recommend.py", "--self-check"]),
          ("CPRI kNN reproduction", ["regression.py", "--cpri"])]


def check_badges():
    """Run every self-check suite now and badge what actually came back.
    These were hard-coded text once, claiming checks that had not run and
    lagging the code (8/8 circuit checks shown against 14 real ones)."""
    import re
    import subprocess
    import sys
    out = []
    for name, args in SUITES:
        r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        n_pass = len(re.findall(r"\[PASS\]", r.stdout))
        n_fail = len(re.findall(r"\[FAIL\]", r.stdout))
        ok = r.returncode == 0 and n_fail == 0 and n_pass > 0
        print(f"  {name}: {n_pass} passed, {n_fail} failed, exit {r.returncode}")
        out.append(f'<span class="badge{" ok" if ok else ""}">'
                   f'{n_pass}/{n_pass + n_fail} {name} checks{"" if ok else " — FAILING"}</span>')
    return "\n      ".join(out)


def build(out_html="impulse_app.html", reuse_session=False):
    fd_path = os.path.join(APPDIR, "frontend_data.json")
    with open(fd_path, encoding="utf-8") as f:
        reco = json.load(f)
    if reco.get("hybrid"):
        check_hybrid_js()
    out_path = os.path.join(HERE, out_html)
    if reuse_session and os.path.exists(out_path):
        print("reusing the illustrative session from the existing app")
        dash = reused_dashboard_data(out_path)
    else:
        dash = dashboard_data()

    print("running self-check suites for the header badges")
    html = TEMPLATE.replace("__RECO__", json.dumps(reco)) \
                    .replace("__DASH__", json.dumps(dash)) \
                    .replace("__BADGES__", check_badges())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


TEMPLATE = open(os.path.join(APPDIR, "app_template.html"), encoding="utf-8").read() \
    if os.path.exists(os.path.join(APPDIR, "app_template.html")) else ""

if __name__ == "__main__":
    import sys
    path = build(reuse_session="--reuse-session" in sys.argv)
    print(f"{path} written")
