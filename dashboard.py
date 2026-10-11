"""
dashboard.py — self-contained HTML: one illustrative test session shot by
shot, plus the aggregate benchmark.

The illustrative session is CHOSEN to show a difference between strategies,
and the page says so. The aggregate benchmark underneath is the evidence;
the session is only there to make it legible. Presenting a hand-picked
session as typical would be the same mistake as reporting an accuracy on
leaked data.

    python dashboard.py            # needs benchmark_summary.csv
"""
import json
import os
import numpy as np
import pandas as pd

from circuit import time_axis
from lab import CATEGORIES
from waveform import TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")


def pick_from_benchmark(results=None, kind="LI"):
    """The illustrative session, taken from the benchmark itself rather than
    a separate search. Declared rule, applied to benchmark_results.csv:
    lightning sessions where lab memory complied, ranked by shots saved over
    current practice — counted as the benchmark counts them, an unsolved
    session at the 10-shot budget — then by resistor changes saved, then by
    severity. The top one is replayed shot by shot, and the replay must
    reproduce that benchmark row for EVERY strategy, or nothing is shown.

    An earlier version also required current practice to comply. On
    lightning it either complies within two shots or not within ten, so that
    rule could only ever show 2 shots against 1.

    The old search (pick_session) drew fresh halls outside the benchmark,
    took ~35 min and found only 2 shots against 1."""
    import benchmark as B
    df = pd.read_csv(results or os.path.join(RESULTS, "benchmark_results.csv"))
    if "session" not in df.columns:
        raise SystemExit("benchmark_results.csv predates the session column; "
                         "re-run python bench_chunk.py --parallel")
    pool = df[(df.feasible_settings > 0) & (df.kind == kind)]
    c = pool[pool.lab_memory_solved]
    eff = np.where(c.technician_solved, c.technician_shots, B.S.MAX_SHOTS)
    c = c.assign(gap=eff - c.lab_memory_shots,
                 dclimb=c.technician_rchanges - c.lab_memory_rchanges)
    row = c.sort_values(["gap", "dclimb", "severity"], ascending=False,
                        kind="stable").iloc[0]
    sev, e = int(row.severity), int(row.session)
    got = {}

    def keep(r, obj, shots):
        if r["session"] == e:
            got.update(row=r, obj=obj, shots=shots)
    B.run([sev], 1, warmup=10, evals=e + 1, halls=[int(row.hall) - 1000 * sev],
          on_session=keep)
    for st in B.STRATS:
        assert got["row"][f"{st}_shots"] == row[f"{st}_shots"], \
            f"replay of the benchmark session disagrees on {st}"
    return got["obj"], got["shots"], row, {
        "candidates": len(c), "sessions": len(pool),
        "tech_unsolved": int((~pool.technician_solved).sum()),
        "tech_max_solved": int(pool[pool.technician_solved].technician_shots.max())}


def session_data(obj, shots, row, counts):
    """The evidence tab's session block, shared by dashboard.py and
    build_app.py so the two can never describe it differently."""
    import benchmark as B
    tech, memr = shots["technician"], shots["lab_memory"]
    tx, tech_curves = curve(tech, obj.kind)
    _, mem_curves = curve(memr, obj.kind)
    T = TARGETS[obj.kind]
    return {
        "session": {
            "object": obj.category.replace("_", " "),
            "c_nominal_pf": round(obj.c_nominal * 1e12, 1),
            "c_cat_range": "%d-%d pF" % CATEGORIES[obj.category][0],
            "target_kv": round(obj.target_v / 1e3),
            "kind": T.get("name", obj.kind),
            "severity": int(row.severity), "hall": int(row.hall),
            "session": int(row.session), **counts,
            "shots": {st: int(row[f"{st}_shots"]) for st in B.STRATS},
            "solved": {st: bool(row[f"{st}_solved"]) for st in B.STRATS},
            "climbs": {st: int(row[f"{st}_rchanges"]) for st in B.STRATS},
            "t1_target": T["t1"], "t1_tol": T["t1_tol"],
            "t2_target": T["t2"], "t2_tol": T["t2_tol"],
        },
        "tech": shot_rows(tech, obj.target_v),
        "mem": shot_rows(memr, obj.target_v),
        "t": tx, "tech_curves": tech_curves, "mem_curves": mem_curves,
    }


def shot_rows(shots, target_v):
    out = []
    for i, s in enumerate(shots, 1):
        r, st = s["result"], s["setting"]
        e = r.errors(target_v)
        out.append({
            "shot": i, "stages": st.n,
            "vcharge_kv": round(st.v_charge / 1e3, 1),
            "rf": st.rf_stage, "rt": st.rt_stage,
            "t1": round(r.t1, 3), "t2": round(r.t2, 2),
            "overshoot": round(100 * r.overshoot, 1),
            "verr": round(100 * e["v"], 2),
            "ok": bool(s["compliant"]),
        })
    return out


def curve(shots, kind):
    """Downsampled normalised test-voltage curves for plotting."""
    t = time_axis(kind) * 1e6
    keep = t <= (100 if kind == "LI" else 6000)
    idx = np.where(keep)[0][::max(1, keep.sum() // 400)]
    out = []
    for s in shots:
        c = s["result"].test_curve
        if c is None:
            continue
        c = np.asarray(c)
        pk = np.max(np.abs(c)) or 1.0
        out.append([round(float(x), 4) for x in (c[idx] / pk)])
    return [round(float(x), 3) for x in t[idx]], out


def build(out_html="impulse_dashboard.html"):
    obj, shots, row, counts = pick_from_benchmark()

    summ_path = os.path.join(RESULTS, "benchmark_summary.csv")
    bench = pd.read_csv(summ_path).to_dict("records") \
        if os.path.exists(summ_path) else []
    meta_path = os.path.join(RESULTS, "benchmark_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) \
        if os.path.exists(meta_path) else {}

    rpath = os.path.join(RESULTS, "regression_results.csv")
    lpath = os.path.join(RESULTS, "regression_learning_curve.csv")
    reg = pd.read_csv(rpath).to_dict("records") if os.path.exists(rpath) else []
    lc = pd.read_csv(lpath).to_dict("records") if os.path.exists(lpath) else []

    data = {
        **session_data(obj, shots, row, counts),
        "bench": bench, "meta": meta, "reg": reg, "lc": lc,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(os.path.join(HERE, out_html), "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(shots["technician"]), len(shots["lab_memory"])


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Impulse Generator Settings</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e2e0;--card:#fff;
--ok:#2d7a4f;--bad:#b23b2e;--a:#1f5fa8;--b:#b8860b;--band:rgba(45,122,79,.09)}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--mut:#9a9a9a;
--line:#2c2c2c;--card:#1c1c1c;--ok:#5cb87f;--bad:#e06a5a;--a:#6da8e8;--b:#d4a537;
--band:rgba(92,184,127,.10)}}
*{box-sizing:border-box}body{margin:0;padding:22px;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
.w{max-width:1180px;margin:0 auto}h1{font-size:17px;margin:0 0 2px;font-weight:600}
.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.p{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:14px 16px;margin-bottom:12px}
.t{font-size:12.5px;font-weight:600}.n{font-size:11.5px;color:var(--mut);margin:2px 0 10px}
.two{display:grid;gap:12px;grid-template-columns:1fr 1fr}@media(max-width:780px){.two{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--mut);font-size:10.5px;
letter-spacing:.02em;padding:5px 6px;border-bottom:1px solid var(--line)}
td{padding:6px;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:none}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad)}
svg{width:100%;height:auto;display:block;overflow:visible}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kv div{font-size:11px;color:var(--mut)}.kv b{display:block;font-size:17px;color:var(--fg);font-weight:600}
.foot{font-size:11px;color:var(--mut);line-height:1.6;margin-top:12px}
.note{border-left:3px solid var(--b);padding:8px 12px;background:var(--card);border-radius:4px;
font-size:12px;margin-bottom:12px}
</style></head><body><div class="w">
<h1>Impulse Generator Settings — 400 kV Insulator Testing</h1>
<div class="sub">Settings recommendation for CPRI's HAEFELY 508085/E — 2400 kV, 12-stage Marx — evaluated per IEC 60060-1</div>

<div class="p"><div class="t">Test request</div><div class="kv" id="req"></div></div>

<div class="note" id="caveat"></div>

<div class="two">
 <div class="p"><div class="t">Current practice — textbook design, then adjust</div>
 <div class="n">Rule of thumb every lab uses: scale Rf by the front-time error, Rt by the tail error.</div>
 <div id="cA"></div><div id="tA"></div></div>
 <div class="p"><div class="t">With lab memory — calibrated from this hall's own history</div>
 <div class="n">First shot already uses what this hall learned: its parasitics and each insulator type's capacitance bias.</div>
 <div id="cB"></div><div id="tB"></div></div>
</div>

<div class="p"><div class="t">Prediction accuracy against CPRI's weighted-KNN correction</div>
<div class="n">The comparison Dr. Sheshu asked for. Each model corrects the textbook prediction of front time, tail time and
peak using this hall's shot history. Error is mean absolute percentage error; pass/fail accuracy is whether the model
tells the operator the right answer on waveshape. Random CV mixes shots freely; the extrapolation test trains on the
smaller 70% of insulators and tests on the larger 30%, which a real lab meets constantly.</div>
<div id="reg"></div>
<div class="two" style="margin-top:12px"><div id="lc1"></div><div id="lc2"></div></div></div>

<div class="p"><div class="t">Benchmark across the severity sweep</div>
<div class="n" id="bn"></div><div class="two"><div id="b1"></div><div id="b2"></div></div>
<div id="btab" style="margin-top:10px"></div></div>

<div class="foot" id="foot"></div></div>
<script>
const D=__DATA__, S=D.session;
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('req').innerHTML=[
 ['Test object',S.object],['Effective load capacitance',S.c_nominal_pf+' pF'],
 ['Unit capacitance, this type',S.c_cat_range],
 ['Waveform',S.kind],['Target test voltage',S.target_kv+' kV'],
 ['Front time window',(S.t1_target*(1-S.t1_tol)).toFixed(2)+'–'+(S.t1_target*(1+S.t1_tol)).toFixed(2)+' µs'],
 ['Tail time window',(S.t2_target*(1-S.t2_tol)).toFixed(0)+'–'+(S.t2_target*(1+S.t2_tol)).toFixed(0)+' µs']
].map(k=>'<div>'+k[0]+'<b>'+esc(k[1])+'</b></div>').join('');

document.getElementById('caveat').innerHTML=
 '<b>This session was selected</b> from the benchmark as the lightning session where lab memory saved the '
 +'most shots over current practice (severity '+S.severity+', hall '+S.hall+', session '+S.session+'). '
 +'Current practice failed to comply within the 10-shot budget in '+S.tech_unsolved+' of the benchmark's '
 +S.sessions+' lightning sessions. '
 +'It is an illustration, not a typical result. The aggregate benchmark below is the evidence, '
 +'including the conditions where current practice already succeeds first time. '
 +'Effective load capacitance includes fixtures and strays and is scaled with setup severity, '
 +'so it exceeds the single-unit range for this insulator type.';

function wave(el,curves,color){
 const W=560,H=190,L=34,R=8,T=8,B=22,t=D.t,n=t.length,tmax=t[n-1];
 const x=v=>L+(v/tmax)*(W-L-R),y=v=>T+(H-T-B)*(1-(v+.05)/1.15);
 let s='<svg viewBox="0 0 '+W+' '+H+'">';
 const f1=S.t1_target*(1-S.t1_tol),f2=S.t1_target*(1+S.t1_tol);
 s+='<rect x="'+x(f1)+'" y="'+T+'" width="'+(x(f2)-x(f1))+'" height="'+(H-T-B)+'" fill="var(--band)"/>';
 const h1=S.t2_target*(1-S.t2_tol),h2=S.t2_target*(1+S.t2_tol);
 s+='<rect x="'+x(h1)+'" y="'+T+'" width="'+(x(h2)-x(h1))+'" height="'+(H-T-B)+'" fill="var(--band)"/>';
 [0,.5,1].forEach(g=>{s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(g)+'" y2="'+y(g)+'" stroke="var(--line)"/>';
  s+='<text x="'+(L-5)+'" y="'+(y(g)+3)+'" text-anchor="end" font-size="9" fill="var(--mut)">'+g+'</text>';});
 curves.forEach((c,i)=>{const last=i===curves.length-1;
  s+='<path d="'+c.map((v,j)=>(j?'L':'M')+x(t[j])+' '+y(v)).join(' ')+'" fill="none" stroke="'+color
   +'" stroke-width="'+(last?1.8:1)+'" opacity="'+(last?1:.35)+'"/>';});
 for(let v=0;v<=tmax;v+=tmax/5)s+='<text x="'+x(v)+'" y="'+(H-6)+'" text-anchor="middle" font-size="9" fill="var(--mut)">'+v.toFixed(0)+' µs</text>';
 s+='</svg><div class="n" style="margin-top:4px">Green bands: IEC front-time and tail-time windows. Faded lines are earlier shots; solid is the last.</div>';
 document.getElementById(el).innerHTML=s;}
function tab(el,rows){
 document.getElementById(el).innerHTML='<table><tr><th>Shot</th><th>Stg</th><th>kV/stg</th><th>Rf</th><th>Rt</th><th>T1 (µs)</th><th>T2 (µs)</th><th>Ovr %</th><th>V err %</th><th></th></tr>'
 +rows.map(r=>'<tr><td>'+r.shot+'</td><td>'+r.stages+'</td><td>'+r.vcharge_kv+'</td><td>'+r.rf+'</td><td>'+r.rt+'</td><td>'
 +r.t1+'</td><td>'+r.t2+'</td><td>'+r.overshoot+'</td><td>'+r.verr+'</td><td class="'+(r.ok?'ok':'bad')+'">'
 +(r.ok?'pass':'fail')+'</td></tr>').join('')+'</table>';}
wave('cA',D.tech_curves,'var(--b)');tab('tA',D.tech);
wave('cB',D.mem_curves,'var(--a)');tab('tB',D.mem);

const B=D.bench;
if(B.length){
 const strats=[['technician','Current practice','var(--b)'],['blackbox_bo','Black-box BO','var(--mut)'],
  ['calibration','Physics calibration','var(--ok)'],['lab_memory','Lab memory','var(--a)'],
  ['app_loop','App shot loop','#8B4A7A'],['app_memory','App with hall memory','#5E6B1F']].filter(st => B.length && (st[0]+'_shots') in B[0]);
 const tot=B.reduce((a,r)=>a+r.sessions,0);
 document.getElementById('bn').textContent='Feasible sessions: '+tot
  +(D.meta.infeasible!=null?' · excluded because no stocked setting can comply: '+D.meta.infeasible+' of '+D.meta.total:'')
  +'. Severity 1 is a well-characterised hall; 4 is long leads, unknown strays, a slow divider and poor capacitance estimates.';
 function line(el,key,title,ymax,fmt){
  const W=540,H=200,L=38,R=10,T=18,Bm=26,sev=B.map(r=>r.severity),n=sev.length;
  const x=i=>L+(n>1?i/(n-1):.5)*(W-L-R),y=v=>T+(H-T-Bm)*(1-v/ymax);
  let s='<svg viewBox="0 0 '+W+' '+H+'"><text x="'+L+'" y="11" font-size="10.5" fill="var(--fg)" font-weight="600">'+title+'</text>';
  for(let g=0;g<=4;g++){const v=ymax*g/4;s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="var(--line)"/>'
   +'<text x="'+(L-5)+'" y="'+(y(v)+3)+'" text-anchor="end" font-size="9" fill="var(--mut)">'+fmt(v)+'</text>';}
  sev.forEach((v,i)=>s+='<text x="'+x(i)+'" y="'+(H-8)+'" text-anchor="middle" font-size="9" fill="var(--mut)">severity '+v+'</text>');
  strats.forEach(st=>{const vals=B.map(r=>r[st[0]+key]);
   s+='<path d="'+vals.map((v,i)=>(i?'L':'M')+x(i)+' '+y(v)).join(' ')+'" fill="none" stroke="'+st[2]+'" stroke-width="1.8"/>';
   vals.forEach((v,i)=>s+='<circle cx="'+x(i)+'" cy="'+y(v)+'" r="2.6" fill="'+st[2]+'"/>');});
  s+='</svg><div class="n">'+strats.map(st=>'<span style="color:'+st[2]+'">&#9644;</span> '+st[1]).join(' &nbsp; ')+'</div>';
  document.getElementById(el).innerHTML=s;}
 line('b1','_first_%','First-shot IEC compliance (%)',100,v=>v.toFixed(0));
 const mx=Math.max(2,...B.flatMap(r=>strats.map(st=>r[st[0]+'_shots']||0)));
 line('b2','_shots','Mean shots to compliance',Math.ceil(mx+.5),v=>v.toFixed(1));
 document.getElementById('btab').innerHTML='<table><tr><th>Severity</th><th>Sessions</th>'
  +strats.map(s=>'<th>'+s[1]+'<br>first % · shots · solved %</th>').join('')+'</tr>'
  +B.map(r=>'<tr><td>'+r.severity+'</td><td>'+r.sessions+'</td>'+strats.map(s=>'<td>'
   +r[s[0]+'_first_%']+' · '+r[s[0]+'_shots']+' · '+r[s[0]+'_solved_%']+'</td>').join('')+'</tr>').join('')+'</table>';
}else document.getElementById('bn').textContent='Run benchmark.py to populate this panel.';

if(D.reg.length){
 const splits=[...new Set(D.reg.map(r=>r.split))];
 document.getElementById('reg').innerHTML=splits.map(sp=>{
  const rows=D.reg.filter(r=>r.split===sp);
  const best=Math.min(...rows.map(r=>r['t1_mape_%']));
  return '<div class="n" style="margin:8px 0 4px;font-weight:600;color:var(--fg)">'+esc(sp)+'</div>'
   +'<table><tr><th>Model</th><th>Front time error</th><th>Tail time error</th><th>Peak error</th><th>Pass/fail correct</th></tr>'
   +rows.map(r=>{const k=r.model.includes('CPRI'),o=r.model.includes('ours');
    return '<tr style="'+(k?'background:var(--band)':'')+'"><td>'+(o?'<b>':'')+esc(r.model)+(o?'</b>':'')
     +'</td><td'+(r['t1_mape_%']===best?' class="ok"':'')+'>'+r['t1_mape_%']+'%</td><td>'+r['t2_mape_%']+'%</td><td>'
     +r['eff_mape_%']+'%</td><td>'+r['compliance_acc_%']+'%</td></tr>';}).join('')+'</table>';}).join('');
}
if(D.lc.length){
 const models=[['Weighted KNN (CPRI baseline)','CPRI weighted KNN','var(--b)'],['Gradient boosting','Gradient boosting','var(--mut)'],
  ['Hybrid: physics + boosting (ours)','Hybrid (ours)','var(--a)']];
 const ns=[...new Set(D.lc.map(r=>r.train_shots))].sort((a,b)=>a-b);
 function lcChart(el,key,title,lo,hi,fmt){
  const W=540,H=210,L=40,R=10,T=18,B=28,x=i=>L+i/(ns.length-1)*(W-L-R),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let s='<svg viewBox="0 0 '+W+' '+H+'"><text x="'+L+'" y="11" font-size="10.5" font-weight="600" fill="var(--fg)">'+title+'</text>';
  for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4;s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="var(--line)"/>'
   +'<text x="'+(L-5)+'" y="'+(y(v)+3)+'" text-anchor="end" font-size="9" fill="var(--mut)">'+fmt(v)+'</text>';}
  ns.forEach((n,i)=>s+='<text x="'+x(i)+'" y="'+(H-10)+'" text-anchor="middle" font-size="9" fill="var(--mut)">'+n+' shots</text>');
  models.forEach(m=>{const v=ns.map(n=>(D.lc.find(r=>r.train_shots===n&&r.model===m[0])||{})[key]);
   s+='<path d="'+v.map((q,i)=>(i?'L':'M')+x(i)+' '+y(q)).join(' ')+'" fill="none" stroke="'+m[2]+'" stroke-width="1.8"/>';
   v.forEach((q,i)=>s+='<circle cx="'+x(i)+'" cy="'+y(q)+'" r="2.6" fill="'+m[2]+'"/>');});
  s+='</svg><div class="n">'+models.map(m=>'<span style="color:'+m[2]+'">&#9644;</span> '+m[1]).join(' &nbsp; ')+'</div>';
  document.getElementById(el).innerHTML=s;}
 lcChart('lc1','t1_mape_%','Front-time error vs shot history (%)',0,7,v=>v.toFixed(1));
 lcChart('lc2','compliance_acc_%','Pass/fail correct vs shot history (%)',75,100,v=>v.toFixed(0));
}

document.getElementById('foot').innerHTML=
 'All results are from a simulated impulse hall with effects the design model does not include — lead inductance, strays, divider response, '
 +'resistor tolerance and test-object capacitance error. They measure robustness to that specified class of mismatch, not performance on a real generator. '
 +'Waveform parameters follow IEC 60060-1:2010 including the test voltage function; confirm which edition CPRI evaluates against.';
</script></body></html>"""


if __name__ == "__main__":
    path, a, b = build()
    print(f"{path} written — illustrative session: current practice {a} shots, "
          f"lab memory {b} shots")
