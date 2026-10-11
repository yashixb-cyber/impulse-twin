"""
atmosphere.py — IEC 60060-1 atmospheric correction.

Standard withstand voltages are defined for a REFERENCE atmosphere: 20 C,
101.3 kPa, 11 g/m3 absolute humidity. A test run at a different temperature,
pressure (or altitude) and humidity must generate a correspondingly
different voltage to represent the same rated withstand level:

    U_test = U_rated * Kt,   Kt = k1 * k2

k1 (air density) and k2 (humidity) each raise a ratio to a gap-geometry
exponent (m for k1, w for k2) that the standard ties to how uniform the
test gap's field is — near-uniform gaps (sphere gaps) get m=w=1 (full
correction), very non-uniform gaps (rod-plane) get less. This tool does not
ask the operator for gap geometry, so it defaults m=w=1.0: the standard's
own value for near-uniform gaps, and the conservative (largest-correction)
choice when the true geometry is unknown. Declared, not confirmed for
CPRI's actual test objects — see CLAUDE.md.

k1 is exact: delta = (p/p0)*(273+t0)/(273+t) is the standard's own
definition, no approximation involved.

k2 is NOT exact here. IEC 60060-1 gives k (humidity factor) as a graph of
k vs h/delta (Figure 2 of the standard), not a formula, and that curve was
not available to digitize precisely. What's implemented is a commonly-cited
engineering approximation: k is linear in h/delta, anchored at k=1 at the
reference point h/delta = 11 g/m3, with slope ~1% per g/m3 — declared
here, in the UI and in CLAUDE.md, exactly as every other unconfirmed
physics value in this project is. Absolute humidity itself (from
temperature and relative humidity) IS an exact formula (ideal-gas-law
psychrometrics, ~ Magnus-Tetens saturation vapour pressure), independent of
HV testing and not an approximation particular to this project.

Altitude-to-pressure (when the operator has an altitude but no barometer)
uses the ICAO standard atmosphere, which assumes a typical lapse rate —
declared as an approximation; enter measured site pressure directly
whenever it is known.
"""
import math
import sys

# Windows consoles default to cp1252 and mangle the em-dashes below.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

T0_C = 20.0            # reference temperature, deg C
P0_KPA = 101.325        # reference pressure, kPa
H0_GM3 = 11.0           # reference absolute humidity, g/m3
K1_RELIABLE = (0.8, 1.05)   # band the standard considers k1 reliable in
HUMIDITY_SLOPE = 0.010      # k2's declared approximation: dk/d(h/delta) per g/m3


def altitude_to_pressure_kpa(altitude_m):
    """ICAO standard atmosphere, troposphere formula. Approximate — real
    site pressure varies with weather; use a measured value if you have one."""
    return P0_KPA * (1 - 2.25577e-5 * altitude_m) ** 5.25588


def absolute_humidity_gm3(temp_c, rh_pct):
    """Absolute humidity from temperature and relative humidity, exactly
    (ideal-gas-law psychrometrics). Saturation vapour pressure via the
    Arden Buck / Magnus-Tetens approximation, valid -40 to 50 deg C."""
    e_sat_hpa = 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))
    e_hpa = (rh_pct / 100.0) * e_sat_hpa
    return 216.7 * e_hpa / (273.15 + temp_c)


def air_density_correction(temp_c, pressure_kpa, m=1.0):
    """(k1, delta). delta = relative air density; k1 = delta**m, the
    standard's own exact definition."""
    delta = (pressure_kpa / P0_KPA) * (273.0 + T0_C) / (273.0 + temp_c)
    return delta ** m, delta


def humidity_correction(abs_humidity_gm3, delta, w=1.0):
    """(k2, h_over_delta). Declared approximation — see module docstring."""
    h_over_delta = abs_humidity_gm3 / delta
    k = 1.0 + HUMIDITY_SLOPE * (h_over_delta - H0_GM3)
    return max(k, 0.0) ** w, h_over_delta


def correction_factor(temp_c=T0_C, rh_pct=None, pressure_kpa=None,
                       altitude_m=None, m=1.0, w=1.0):
    """Kt = k1*k2 for the given site conditions. Every argument defaults to
    the reference atmosphere, so calling this with nothing gives Kt=1.0 —
    the no-correction case recommend.py and the app fall back to when the
    operator hasn't entered site conditions.

    Pass pressure_kpa when known; altitude_m is used only if pressure_kpa
    is not given, via the approximate ICAO conversion above."""
    if rh_pct is None:
        rh_pct = _rh_for_reference_humidity(temp_c)
    if pressure_kpa is None:
        pressure_kpa = P0_KPA if altitude_m is None else altitude_to_pressure_kpa(altitude_m)
    abs_h = absolute_humidity_gm3(temp_c, rh_pct)
    k1, delta = air_density_correction(temp_c, pressure_kpa, m=m)
    k2, h_over_delta = humidity_correction(abs_h, delta, w=w)
    return {
        "kt": k1 * k2, "k1": k1, "k2": k2, "delta": delta,
        "absolute_humidity_gm3": abs_h, "h_over_delta": h_over_delta,
        "k1_reliable": K1_RELIABLE[0] < k1 < K1_RELIABLE[1],
        "pressure_kpa": pressure_kpa, "temp_c": temp_c, "rh_pct": rh_pct,
    }


def _rh_for_reference_humidity(temp_c):
    """RH% at this temperature that gives exactly the reference 11 g/m3 —
    used only so correction_factor(temp_c=X) alone (no humidity given)
    still returns k2=1 rather than silently assuming some arbitrary RH."""
    e_sat_hpa = 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))
    e_needed_hpa = H0_GM3 * (273.15 + temp_c) / 216.7
    return 100.0 * e_needed_hpa / e_sat_hpa


if __name__ == "__main__":
    ok = True

    def chk(name, cond):
        global ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("Reference atmosphere and defaults")
    r = correction_factor()
    chk(f"no arguments -> Kt = 1.0 exactly  (got {r['kt']:.6f})",
        abs(r["kt"] - 1.0) < 1e-9)
    r = correction_factor(temp_c=20.0, rh_pct=_rh_for_reference_humidity(20.0),
                          pressure_kpa=P0_KPA)
    chk(f"explicit reference atmosphere -> Kt = 1.0  (got {r['kt']:.6f})",
        abs(r["kt"] - 1.0) < 1e-6)
    chk("k1 reliable at the reference point", r["k1_reliable"])

    print("\nAir density correction (k1) — exact")
    k1_hot, _ = air_density_correction(45.0, P0_KPA)
    k1_ref, _ = air_density_correction(20.0, P0_KPA)
    chk("hotter air at the same pressure is less dense -> lower k1",
        k1_hot < k1_ref)
    k1_high, _ = air_density_correction(20.0, altitude_to_pressure_kpa(2000))
    chk("lower pressure (altitude) -> lower k1", k1_high < k1_ref)
    chk("k1 unreliable flagged outside the standard's 0.8-1.05 band",
        not (K1_RELIABLE[0] < air_density_correction(20.0, 60.0)[0] < K1_RELIABLE[1]))

    print("\nAltitude -> pressure (ICAO standard atmosphere, approximate)")
    chk("sea level -> reference pressure exactly",
        abs(altitude_to_pressure_kpa(0) - P0_KPA) < 1e-9)
    chk("2000 m -> roughly 79 kPa (textbook ballpark)",
        75.0 < altitude_to_pressure_kpa(2000) < 82.0)

    print("\nAbsolute humidity (exact psychrometrics)")
    h_dry = absolute_humidity_gm3(20.0, 0.0)
    h_sat = absolute_humidity_gm3(20.0, 100.0)
    chk("0% RH -> 0 g/m3", abs(h_dry) < 1e-9)
    chk(f"100% RH at 20 C is a plausible 10-20 g/m3  (got {h_sat:.2f})",
        10.0 < h_sat < 20.0)
    chk("humidity increases monotonically with RH",
        absolute_humidity_gm3(20, 30) < absolute_humidity_gm3(20, 70))
    chk("reference RH helper actually gives 11 g/m3 back",
        abs(absolute_humidity_gm3(20.0, _rh_for_reference_humidity(20.0)) - H0_GM3) < 1e-6)

    print("\nHumidity correction (k2) — declared approximation")
    k2_dry, _ = humidity_correction(2.0, 1.0)
    k2_wet, _ = humidity_correction(20.0, 1.0)
    chk("drier than reference -> k2 < 1", k2_dry < 1.0)
    chk("more humid than reference -> k2 > 1", k2_wet > 1.0)

    print("\nWorked example: hot, dry, moderate altitude (plausibility only —")
    print("our formulas need not match any other implementation's exact number)")
    hot = correction_factor(temp_c=45.0, rh_pct=20.0, altitude_m=380.0)
    chk(f"Kt meaningfully below 1 in hot/dry/high conditions  (got {hot['kt']:.3f})",
        0.80 < hot["kt"] < 0.97)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    raise SystemExit(0 if ok else 1)
