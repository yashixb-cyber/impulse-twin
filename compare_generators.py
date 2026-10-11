"""
compare_generators.py — the same test request on both of CPRI's descriptions
of its generator, side by side. The nameplate (page 4 of the problem
statement: 12 stages, 0.125 uF/stage) and the Excel calculator (15 stages,
3 uF/stage, its own resistor inventory) disagree and CPRI has not said which
is real. Generator is a parameter everywhere, so both can be run.

    python compare_generators.py [--level 1550] [--c-test 90]

Lightning only: the Excel inventory has one tail list and no switching tail,
and its own dataset's switching settings are not buildable from either stock
(CLAUDE.md P6).
"""
import argparse
import sys
import time

import recommend as RC
from circuit import Generator

EXCEL_FRONT = RC.CPRI_EXCEL_STOCK["front"]
EXCEL_TAIL = RC.CPRI_EXCEL_STOCK["tail"]

GENERATORS = {
    "Nameplate (12 stages, 0.125 uF)": Generator(),
    "Excel calculator (15 stages, 3 uF)": Generator(
        n_stages=15, c_stage=3e-6, front_units=EXCEL_FRONT,
        tail_li_units=EXCEL_TAIL, tail_si_units=EXCEL_TAIL),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1550, help="LI withstand, kV")
    ap.add_argument("--um", type=int, default=420)
    ap.add_argument("--c-test", type=float, default=90.0, help="pF")
    a = ap.parse_args()
    print(f"Lightning impulse, {a.level} kV, {a.c_test:.0f} pF test object\n")
    for name, g in GENERATORS.items():
        t0 = time.time()
        try:
            rec = RC.recommend("porcelain_disc_string", a.um, a.level, "LI",
                               c_test_pf=a.c_test, top=3, g=g)
        except RC.InputError as e:
            print(f"{name}: {e}\n")
            continue
        print(f"{name}   ({time.time() - t0:.0f} s search)")
        for d in rec["options"]:
            s = d["setting"]
            print(f"  {s.n:2d} stages  Rf {s.rf_stage:6.0f} ohm  Rt {s.rt_stage:6.0f} ohm"
                  f"  {s.v_charge / 1e3:6.1f} kV/stage ({d['charge_frac'] * 100:3.0f}%)"
                  f"  T1 {d['t1']:.2f} us  T2 {d['t2']:.1f} us  os {d['overshoot'] * 100:.1f}%")
        print()
    print("Both are self-consistent: tail resistance per stage = T2/(0.7 * C_stage)"
          " gives\n  571 ohm for 0.125 uF and 24 ohm for 3 uF. Physics alone cannot"
          " choose between them;\n  the nameplate photograph is the physical evidence.")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
