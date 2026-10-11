"""
recommend.py — the operator-facing entry point.

The briefing is explicit that the user is a layman: they know the test
object's insulation level — "1550 kV lightning impulse for a 400 kV
insulator" — and they do NOT know stray inductance or load capacitance.
Everything below that line is the tool's job.

So this module takes what a test request actually contains:

    equipment type, system voltage class (Um), withstand level, impulse type

and returns what the operator has to set on the generator: stage count,
front resistor, tail resistor, charging voltage, and any added load
capacitor — with the test-object capacitance INFERRED from the equipment
type rather than demanded from the user.

Wrong inputs are caught here, which the briefing also asks for. A level
that is not an IEC 60071 standard value for the declared class is rejected
with the valid options listed; a level that is standard but unusual for the
declared equipment is allowed with a warning, because labs do test off the
usual combination and refusing would be wrong.

UNCONFIRMED — the standard levels below are IEC 60071-1 as we read it, and
CLAUDE.md P6 flags them for confirmation against the edition CPRI uses. The
420 kV row is the one this project turns on. Confirm before the demo.
"""
import argparse
import os
import sys

import numpy as np

import atmosphere
from circuit import Generator, resistor_options
from design import Belief, design

# Windows consoles default to cp1252 and mangle the em-dashes below. This is
# the operator-facing tool, so it has to read cleanly wherever it is run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# IEC 60071-1 standard withstand voltages, kV peak, by Um class.
#   Um: (lightning impulse levels, switching impulse levels)
# Range I (Um <= 245 kV) has no standard switching-impulse level: the
# lightning impulse governs, which is why those rows carry an empty tuple.
IEC_60071 = {
    145: ((450, 550, 650), ()),
    245: ((750, 850, 950, 1050), ()),
    362: ((950, 1050, 1175), (850, 950)),
    420: ((1050, 1175, 1300, 1425, 1550), (850, 950, 1050)),
    550: ((1175, 1300, 1425, 1550, 1675, 1800), (1050, 1175)),
    800: ((1675, 1800, 1950, 2100), (1300, 1425, 1550)),
}

# Test-object capacitance by equipment type, pF. This is the PUBLIC nominal
# an operator could read off a datasheet — deliberately duplicated rather
# than imported from lab.py, because lab.py is the hidden hall and the
# design side must not see into it. The lab's per-type estimate BIAS stays
# hidden there; only the nominal range is knowable here.
EQUIPMENT = {
    "porcelain_disc_string": ((60, 120), "Porcelain cap-and-pin disc string"),
    "composite_long_rod": ((30, 60), "Composite (silicone) long rod"),
    "post_insulator_stack": ((120, 250), "Station post insulator stack"),
    "bushing": ((150, 400), "Transformer or wall bushing"),
}

# Which Um class each equipment type is normally tested at. Used only to
# warn, never to reject.
TYPICAL_UM = {"porcelain_disc_string": (245, 420), "composite_long_rod": (245, 420),
              "post_insulator_stack": (420, 550), "bushing": (420, 550, 800)}


class InputError(ValueError):
    """A test request that cannot be honoured as written."""


def validate(equipment, um_kv, level_kv, impulse):
    """Check a request against IEC 60071. Returns a list of warnings, or
    raises InputError with the valid options when the request is not
    standard. Warnings are for choices that are legal but unusual."""
    warnings = []
    if equipment not in EQUIPMENT:
        raise InputError(
            f"unknown equipment type {equipment!r}. "
            f"Known types: {', '.join(sorted(EQUIPMENT))}")
    if impulse not in ("LI", "SI"):
        raise InputError(f"impulse must be LI or SI, not {impulse!r}")
    if um_kv not in IEC_60071:
        raise InputError(
            f"{um_kv} kV is not an IEC 60071 system voltage class. "
            f"Standard classes: {', '.join(str(u) for u in sorted(IEC_60071))}")

    li_levels, si_levels = IEC_60071[um_kv]
    levels = li_levels if impulse == "LI" else si_levels
    if not levels:
        raise InputError(
            f"IEC 60071 defines no standard switching-impulse level for the "
            f"{um_kv} kV class (range I — the lightning impulse governs). "
            f"Test at lightning impulse instead: "
            f"{', '.join(str(v) for v in li_levels)} kV.")
    if level_kv not in levels:
        raise InputError(
            f"{level_kv} kV is not a standard {impulse} withstand level for "
            f"the {um_kv} kV class. Standard levels: "
            f"{', '.join(str(v) for v in levels)} kV.")

    if um_kv not in TYPICAL_UM[equipment]:
        warnings.append(
            f"{EQUIPMENT[equipment][1]} is not usually tested at the "
            f"{um_kv} kV class — proceeding, but confirm the request.")
    return warnings


def infer_capacitance(equipment, c_test_pf=None):
    """The number the operator does not have. Midpoint of the type's range
    unless they measured it. Returned with the range, because the spread is
    what the robustness margin is defending against."""
    lo, hi = EQUIPMENT[equipment][0]
    if c_test_pf is not None:
        if not lo * 0.3 <= c_test_pf <= hi * 3:
            raise InputError(
                f"{c_test_pf} pF is far outside the {lo}-{hi} pF range for "
                f"{EQUIPMENT[equipment][1]}. Check the units (pF, not nF).")
        return float(c_test_pf), (lo, hi), True
    return (lo + hi) / 2.0, (lo, hi), False


SITE_LIMITS = {"temperature": ("temp_c", -40, 60, "°C"), "relative humidity": ("rh_pct", 0, 100, "%"),
               "site pressure": ("pressure_kpa", 30, 110, "kPa"), "altitude": ("altitude_m", -500, 9000, "m")}


def check_site(**site):
    """Reject non-physical site conditions. A pressure of 0 kPa used to give
    Kt = NaN, and RH 200% was silently accepted."""
    for label, (key, lo, hi, unit) in SITE_LIMITS.items():
        v = site.get(key)
        if v is not None and not (lo <= v <= hi):
            raise InputError(f"{label} {v:g} {unit} is outside {lo} to {hi} {unit}, the "
                             f"range atmospheric correction is defined for. Check the entry.")


POLARITIES = ("positive", "negative")
LOOP_UH_RANGE = (1.0, 120.0)   # external loop inductance the model is validated for, uH


def recommend(equipment, um_kv, level_kv, impulse, c_test_pf=None, top=3,
              g: Generator = None, temp_c=None, rh_pct=None,
              pressure_kpa=None, altitude_m=None, polarity="positive",
              parallel=(1,), loop_inductance_uh=None):
    """Full recommendation for one test request. Raises InputError on a
    request that is not standard. Series/parallel resistor pairs are always
    searched: on the nameplate stock they are how a 1.2 us front is made.

    Atmospheric correction (IEC 60060-1, atmosphere.py) is applied to the
    target voltage when site conditions are given; leaving temp_c (and
    everything else atmospheric) at None means Kt=1 — the standard
    reference atmosphere, i.e. no correction, exactly today's behaviour."""
    g = g or Generator()
    belief = Belief()
    if loop_inductance_uh is not None:
        # External lead / loop inductance, replacing the nominal g.l_ext
        # (5 uH). The generator's own 2.5 uH per stage is added separately.
        lo, hi = LOOP_UH_RANGE
        if not (lo <= loop_inductance_uh <= hi):
            raise InputError(
                f"loop inductance {loop_inductance_uh} uH is outside "
                f"{lo}-{hi} uH, the range the model is validated for.")
        belief = Belief(l_extra=loop_inductance_uh * 1e-6 - g.l_ext)
    if polarity not in POLARITIES:
        raise InputError(f"polarity must be one of {', '.join(POLARITIES)}, "
                         f"not {polarity!r}")
    warnings = validate(equipment, um_kv, level_kv, impulse)
    c_test, c_range, measured = infer_capacitance(equipment, c_test_pf)

    check_site(temp_c=temp_c, rh_pct=rh_pct, pressure_kpa=pressure_kpa,
               altitude_m=altitude_m)
    atmo = atmosphere.correction_factor(
        **{k: v for k, v in dict(temp_c=temp_c, rh_pct=rh_pct,
                                 pressure_kpa=pressure_kpa,
                                 altitude_m=altitude_m).items()
           if v is not None})
    target_v = level_kv * 1e3 * atmo["kt"]
    if not atmo["k1_reliable"]:
        warnings.append(
            f"air density correction k1={atmo['k1']:.3f} is outside the "
            f"range IEC 60060-1 considers reliable (0.8-1.05) — treat the "
            f"corrected target voltage with extra caution.")

    options = design(impulse, target_v, c_test * 1e-12, belief, g, top=top,
                     parallel=parallel)
    for d in options:
        d["loading"] = capacitor_loading(d["setting"], g)
    # defence in depth: the search already enforces the 90% operating limit
    options = [d for d in options if d["loading"]["ok"]]
    if not options:
        raise InputError(
            f"no setting in stock produces a compliant {impulse} waveform at "
            f"{level_kv} kV into {c_test:.0f} pF. The target may exceed what "
            f"this generator and resistor stock can reach.")
    return {"options": options, "warnings": warnings, "c_test_pf": c_test,
            "c_range_pf": c_range, "c_measured": measured, "generator": g,
            "equipment": equipment, "um_kv": um_kv, "level_kv": level_kv,
            "impulse": impulse, "atmo": atmo, "target_v": target_v,
            "polarity": polarity, "loop_uh": loop_inductance_uh}


# CPRI's own calculator stock: data/cpri_optimiser.xlsx, "Resistor Inventory"
# (four units of every value per stage). A second description of equipment, next
# to the nameplate's; the self-check below reads the sheet and compares.
CPRI_EXCEL_STOCK = {
    "front": (10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
    "tail": (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200),
    "units": 4,
}
BUILD_TOL = 0.05     # within the +-5% unit tolerance counts as the same value


def capacitor_loading(s, g):
    """Is any stage capacitor asked for more than it is rated? Voltage and
    energy per stage against the nameplate (200 kV, 2.5 kJ), total stored
    energy against 30 kJ. `ok` is the operating limit (90% of rating, what
    every search already enforces); `overload` is the rating itself."""
    e_rated = 0.5 * g.c_stage * g.v_stage_max ** 2
    e_stage = 0.5 * g.c_stage * s.v_charge ** 2
    return {"v_frac": s.v_charge / g.v_stage_max,
            "e_stage_j": e_stage, "e_rated_j": e_rated,
            "e_total_j": s.n * s.p * e_stage, "e_total_rated_j": g.n_stages * e_rated,
            "output_v": s.n * s.v_charge,
            "ok": s.v_charge <= g.charge_margin * g.v_stage_max + 1e-9,
            "overload": s.v_charge > g.v_stage_max + 1e-9}


def excel_reach(value, units, max_units=None):
    """Closest value buildable per stage from CPRI's calculator stock: up to
    `max_units` units in pure series or pure parallel. Returns
    (achieved ohm, error fraction, how)."""
    import itertools
    max_units = max_units or CPRI_EXCEL_STOCK["units"]
    best = None
    for k in range(1, max_units + 1):
        for combo in itertools.combinations_with_replacement(units, k):
            label = " + ".join(f"{combo.count(v)} x {v:g} ohm" for v in sorted(set(combo)))
            cands = [(sum(combo), f"{label} in series" if k > 1 else f"1 x {combo[0]:g} ohm")]
            if k > 1:
                cands.append((1.0 / sum(1.0 / v for v in combo), f"{label} in parallel"))
            for val, how in cands:
                err = abs(val / value - 1)
                if best is None or err < best[1] - 1e-12:
                    best = (val, err, how)
    return best


def _realize(value, stock):
    """How to physically build a chosen resistance from the physical units:
    a single unit if it already is one, otherwise the series/parallel pair.
    Round-trips through resistor_options() at the same precision the stock
    was built with, so the lookup always hits."""
    return resistor_options(stock).get(
        round(value, 2 - int(np.floor(np.log10(value)))) if value > 0 else 0.0,
        f"{value:g} ohm (unrecognised — not from the declared stock)")


def report(rec):
    """Plain-language settings sheet — what the operator sets on the machine."""
    g, imp = rec["generator"], rec["impulse"]
    shape = "1.2/50 us lightning" if imp == "LI" else "250/2500 us switching"
    L = []
    L.append("=" * 68)
    L.append(f"  {EQUIPMENT[rec['equipment']][1]}")
    L.append(f"  {rec['um_kv']} kV class  |  {rec['level_kv']} kV {imp} "
             f"withstand  |  {shape} impulse  |  {rec.get('polarity', 'positive')} polarity")
    L.append("=" * 68)
    for w in rec["warnings"]:
        L.append(f"  WARNING: {w}")
    src = "measured" if rec["c_measured"] else \
        (f"assumed from equipment type, range "
         f"{rec['c_range_pf'][0]}-{rec['c_range_pf'][1]} pF")
    L.append(f"  Test-object capacitance: {rec['c_test_pf']:.0f} pF ({src})")
    atmo = rec["atmo"]
    if abs(atmo["kt"] - 1.0) > 1e-9:
        L.append(f"  Atmospheric correction: Kt={atmo['kt']:.3f} (k1={atmo['k1']:.3f} "
                 f"air density, k2={atmo['k2']:.3f} humidity, both APPROXIMATE — "
                 f"see CLAUDE.md) -> corrected target {rec['target_v'] / 1e3:.1f} kV")
    L.append("")

    for i, d in enumerate(rec["options"], 1):
        s = d["setting"]
        tag = "RECOMMENDED" if i == 1 else f"alternative {i - 1}"
        L.append(f"  [{tag}]")
        conn = (f"{s.n} in series" if s.p == 1 else
                f"{s.p} parallel columns x {s.n} in series")
        L.append(f"     Stages to erect .......... {s.n * s.p} of {g.n_stages}"
                 f"   ({conn})")
        L.append(f"     Charging voltage ......... {s.v_charge / 1e3:.1f} kV "
                 f"per stage  ({d['charge_frac'] * 100:.0f}% of rating)")
        rf_stock = g.front_units
        rt_stock = g.tail_li_units if imp == "LI" else g.tail_si_units
        L.append(f"     Front resistor Rf ........ {s.rf_stage:g} ohm per stage"
                 f"   ({s.n * s.rf_stage / s.p:g} ohm total)")
        L.append(f"        built as: {_realize(s.rf_stage, rf_stock)}")
        L.append(f"     Tail resistor Rt ......... {s.rt_stage:g} ohm per stage"
                 f"   ({s.n * s.rt_stage / s.p:g} ohm total)")
        L.append(f"        built as: {_realize(s.rt_stage, rt_stock)}")
        if imp == "SI":
            L.append(f"     Added load capacitor ..... {s.c_load * 1e9:.0f} nF")
        L.append(f"     Expected waveform ........ {d['t1']:.2f} / {d['t2']:.1f} us"
                 + (f"   overshoot {d['overshoot'] * 100:.1f}%" if imp == "LI" else ""))
        L.append(f"     Charging voltage total ... {s.n * s.v_charge / 1e3:.0f} kV")
        ld = d["loading"]
        L.append(f"     Capacitor loading ........ {ld['v_frac'] * 100:.0f}% of the {g.v_stage_max / 1e3:.0f} kV rating, "
                 f"{ld['e_stage_j'] / 1e3:.2f} of {ld['e_rated_j'] / 1e3:.1f} kJ per stage, "
                 f"{ld['e_total_j'] / 1e3:.1f} of {ld['e_total_rated_j'] / 1e3:.0f} kJ total  "
                 f"[{'OK' if ld['ok'] else 'OVER LIMIT'}]")
        for nm, val, ex in (("front", s.rf_stage, CPRI_EXCEL_STOCK["front"]),
                            ("tail", s.rt_stage, CPRI_EXCEL_STOCK["tail"])):
            v, err, how = excel_reach(val, ex)
            L.append(f"     CPRI Excel stock, {nm} ..... "
                     + (f"{how}" if err <= BUILD_TOL else
                        f"not reachable; nearest {v:g} ohm ({how}, {err * 100:+.0f}%)"))
        L.append(f"     Tolerance margin ......... {d['shape_margin']:.2f} "
                 f"(below 1.00 = inside every IEC band)")
        L.append("")

    L.append(f"  Output assumed: {g.c_base_load * 1e12:.0f} pF basic load, "
             f"{g.c_divider * 1e12:.0f} pF damped capacitive divider,")
    L.append(f"  {g.c_stray_nominal * 1e12:.0f} pF stray to ground. A first shot "
             "at reduced voltage is still")
    L.append("  expected — these settings are the starting point, not a "
             "substitute for it.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def _checks():
    """One runnable check over the validation logic and the happy path."""
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def raises(fn, needle):
        try:
            fn()
            return False
        except InputError as e:
            return needle.lower() in str(e).lower()

    print("Input validation — the briefing asks for wrong inputs to be caught")
    chk("unknown equipment type rejected",
        raises(lambda: validate("banana", 420, 1550, "LI"), "unknown equipment"))
    chk("non-standard system class rejected",
        raises(lambda: validate("porcelain_disc_string", 400, 1550, "LI"),
               "not an IEC 60071 system voltage class"))
    chk("non-standard withstand level rejected, options listed",
        raises(lambda: validate("porcelain_disc_string", 420, 1600, "LI"),
               "standard levels"))
    chk("switching impulse on a range I class rejected",
        raises(lambda: validate("porcelain_disc_string", 245, 950, "SI"),
               "no standard switching-impulse level"))
    chk("bad impulse type rejected",
        raises(lambda: validate("porcelain_disc_string", 420, 1550, "XX"),
               "must be LI or SI"))
    chk("capacitance in nF instead of pF caught",
        raises(lambda: infer_capacitance("porcelain_disc_string", 0.09),
               "check the units"))
    chk("non-physical site pressure (0 kPa, once NaN) rejected",
        raises(lambda: check_site(pressure_kpa=0), "site pressure"))
    chk("unknown polarity rejected",
        raises(lambda: recommend("porcelain_disc_string", 145, 450, "LI",
                                 polarity="sideways"), "polarity"))
    chk("valid 420 kV LI request passes with no warning",
        validate("porcelain_disc_string", 420, 1550, "LI") == [])
    chk("legal but unusual equipment/class pair warns, does not reject",
        len(validate("composite_long_rod", 800, 2100, "LI")) == 1)

    print("\nRecommendation — the briefing's own example")
    r = recommend("porcelain_disc_string", 420, 1550, "LI")
    d = r["options"][0]
    s = d["setting"]
    chk("capacitance inferred, operator never asked for farads",
        r["c_test_pf"] == 90.0 and not r["c_measured"])
    chk(f"front time in the IEC band  ({d['t1']:.2f} us)", 0.84 <= d["t1"] <= 1.56)
    chk(f"tail time in the IEC band  ({d['t2']:.1f} us)", 40.0 <= d["t2"] <= 60.0)
    chk(f"charging voltage within rating  ({d['charge_frac'] * 100:.0f}%)",
        0.0 < d["charge_frac"] <= 0.90)
    chk(f"resistors come from stock  (Rf={s.rf_stage:g}, Rt={s.rt_stage:g})",
        s.rf_stage in Generator().rf_li and s.rt_stage in Generator().rt_li)

    r2 = recommend("post_insulator_stack", 420, 1050, "SI")
    d2 = r2["options"][0]
    chk(f"switching impulse: time to peak in band  ({d2['t1']:.0f} us)",
        200.0 <= d2["t1"] <= 300.0)
    chk(f"switching impulse: tail in band  ({d2['t2']:.0f} us)",
        1000.0 <= d2["t2"] <= 4000.0)

    print("\nPolarity and stage connection")
    rp = recommend("composite_long_rod", 145, 450, "LI", polarity="positive")
    rn = recommend("composite_long_rod", 145, 450, "LI", polarity="negative")
    chk("negative polarity: same setting (the circuit is linear, v -> -v)",
        rp["options"][0]["setting"] == rn["options"][0]["setting"] and rn["polarity"] == "negative")
    rows = design("LI", 450e3, 45e-12, Belief(), Generator(), top=200,
                  restrict=lambda n, rf, rt, cl: n == 6, parallel=(2,))
    chk("2 parallel x 6 series is a reachable connection at 450 kV",
        any(d["setting"].p == 2 for d in rows))
    chk("... but not at 1550 kV (6 x 180 kV = 1080 kV maximum)",
        not any(d["setting"].p == 2 for d in design(
            "LI", 1550e3, 90e-12, Belief(), Generator(), top=200,
            restrict=lambda n, rf, rt, cl: n == 6, parallel=(2,))))

    print("\nCapacitor loading and buildability")
    from circuit import Setting as _S
    gg = Generator()
    ld = d["loading"]
    chk(f"recommended setting loads capacitors within 90%  ({ld['v_frac'] * 100:.0f}%, "
        f"{ld['e_stage_j'] / 1e3:.2f} kJ of {ld['e_rated_j'] / 1e3:.1f} kJ per stage)",
        ld["ok"] and not ld["overload"] and ld["e_stage_j"] <= ld["e_rated_j"])
    chk("a 195 kV/stage charge is flagged over the 90% operating limit",
        not capacitor_loading(_S(n=12, v_charge=195e3, rf_stage=30, rt_stage=520), gg)["ok"])
    chk("a 210 kV/stage charge is flagged as exceeding the rating",
        capacitor_loading(_S(n=12, v_charge=210e3, rf_stage=30, rt_stage=520), gg)["overload"])
    chk("energy per stage at 200 kV is the nameplate's 2.5 kJ",
        abs(capacitor_loading(_S(n=12, v_charge=200e3, rf_stage=30, rt_stage=520), gg)["e_stage_j"] - 2500) < 1)
    chk("every option of every example request is within the operating limit",
        all(o["loading"]["ok"] for rr in (r, r2) for o in rr["options"]))
    v, err, how = excel_reach(30, CPRI_EXCEL_STOCK["front"])
    chk(f"CPRI Excel stock: 30 ohm front is one unit  ({how})", err < 1e-9 and v == 30)
    v, err, how = excel_reach(520, CPRI_EXCEL_STOCK["tail"])
    chk(f"CPRI Excel stock: 520 ohm tail is reachable with four units  ({how})", err <= BUILD_TOL)
    v, err, how = excel_reach(22000, CPRI_EXCEL_STOCK["tail"])
    chk(f"CPRI Excel stock: a 22 kohm switching tail is not reachable  (nearest {v:g} ohm)",
        err > BUILD_TOL)
    try:
        import pandas as _pd
        inv = _pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "data", "cpri_optimiser.xlsx"),
                             sheet_name="Resistor Inventory", header=2)
        front = tuple(int(x) for x in inv.iloc[:, 0].dropna())
        tail = tuple(int(x) for x in inv.iloc[:, 4].dropna())
        chk("stock constants match CPRI's Resistor Inventory sheet",
            front == CPRI_EXCEL_STOCK["front"] and tail == CPRI_EXCEL_STOCK["tail"])
    except Exception as e:
        print(f"  [skipped] could not read the Excel inventory ({type(e).__name__})")

    print("\nLoop inductance input")
    from design import predict as _pred
    from circuit import Setting as _S2
    gg2 = Generator()
    s0 = _S2(n=12, v_charge=1.0, rf_stage=30, rt_stage=520)
    a, _, _ = _pred(s0, gg2, 90e-12, Belief(), "LI")
    b2, _, _ = _pred(s0, gg2, 90e-12, Belief(l_extra=45e-6), "LI")
    # In the underdamped regime inductance first SPEEDS the measured front
    # (the record rings), then slows it; the dependable effect is ringing.
    chk(f"50 uH of loop inductance moves the front time by over 5% ({a.t1:.2f} -> {b2.t1:.2f} us)",
        abs(b2.t1 / a.t1 - 1) > 0.05)
    chk(f"... and raises the overshoot ({a.overshoot * 100:.1f} -> {b2.overshoot * 100:.1f} %)",
        b2.overshoot > a.overshoot)
    base = recommend("porcelain_disc_string", 420, 1550, "LI")
    same = recommend("porcelain_disc_string", 420, 1550, "LI", loop_inductance_uh=5.0)
    chk("the nominal 5 uH reproduces the default recommendation exactly",
        base["options"][0]["setting"] == same["options"][0]["setting"])
    chk("loop inductance outside 1-120 uH is rejected",
        raises(lambda: recommend("porcelain_disc_string", 420, 1550, "LI",
                                 loop_inductance_uh=500), "loop inductance")
        and raises(lambda: recommend("porcelain_disc_string", 420, 1550, "LI",
                                     loop_inductance_uh=-3), "loop inductance"))

    print("\nAtmospheric correction (IEC 60060-1, atmosphere.py)")
    chk("no site conditions given -> unchanged target voltage (Kt=1)",
        r["target_v"] == 1550e3 and r["atmo"]["kt"] == 1.0)
    r3 = recommend("porcelain_disc_string", 420, 1550, "LI",
                   temp_c=45.0, rh_pct=20.0, altitude_m=380.0)
    chk(f"hot/dry/high site lowers the target voltage  ({r3['target_v']/1e3:.0f} kV "
        f"vs 1550 kV rated, Kt={r3['atmo']['kt']:.3f})",
        r3["target_v"] < 1550e3 and r3["atmo"]["kt"] < 1.0)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Recommend impulse generator settings from a test request.")
    ap.add_argument("--equipment", default="porcelain_disc_string",
                    choices=sorted(EQUIPMENT))
    ap.add_argument("--um", type=int, default=420,
                    help="system voltage class Um, kV (e.g. 420)")
    ap.add_argument("--level", type=int, default=1550,
                    help="withstand level, kV peak (e.g. 1550)")
    ap.add_argument("--impulse", default="LI", choices=("LI", "SI"))
    ap.add_argument("--c-test", type=float, default=None,
                    help="measured test-object capacitance in pF, if known")
    ap.add_argument("--temp-c", type=float, default=None,
                    help="site temperature, deg C (enables atmospheric correction)")
    ap.add_argument("--rh", type=float, default=None,
                    help="site relative humidity, %% (default: reference humidity)")
    ap.add_argument("--pressure-kpa", type=float, default=None,
                    help="site pressure, kPa (preferred over --altitude-m)")
    ap.add_argument("--altitude-m", type=float, default=None,
                    help="site altitude, m (approximate pressure if no barometer)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        sys.exit(0 if _checks() else 1)
    try:
        print(report(recommend(a.equipment, a.um, a.level, a.impulse,
                               a.c_test, temp_c=a.temp_c, rh_pct=a.rh,
                               pressure_kpa=a.pressure_kpa,
                               altitude_m=a.altitude_m)))
    except InputError as e:
        raise SystemExit(f"\n  Cannot run this test as requested:\n    {e}\n")
