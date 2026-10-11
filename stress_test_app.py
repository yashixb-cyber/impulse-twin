"""
stress_test_app.py — drive impulse_app.html the way a hostile judge would.

    python stress_test_app.py

Every equipment type x impulse kind x system voltage x withstand level, wrong
and extreme capacitance / atmosphere / shot-loop entries, localStorage that
throws, light and dark. Fails on any uncaught page error, console error, or
NaN / undefined / Infinity text in a result. Needs Playwright with Chromium.
"""
import itertools
import os
import pathlib
import re
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
APP = (HERE / "impulse_app.html").resolve().as_uri()
BAD = re.compile(r"NaN|undefined|Infinity|\[object")
problems, checks = [], 0


def chrome_path():
    base = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    hits = sorted(base.glob("chromium-*/chrome-win64/chrome.exe"))
    return str(hits[-1]) if hits else None


def new_page(b, **kw):
    ctx = b.new_context(**kw)
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errs.append(f"console.{m.type}: {m.text}")
          if m.type == "error" and "fonts.g" not in m.text and "ERR_" not in m.text else None)
    pg.goto(APP)
    pg.wait_for_timeout(300)
    return ctx, pg, errs


def check(label, pg, errs, out="#reco-out"):
    global checks
    checks += 1
    txt = pg.inner_text(out) if pg.query_selector(out) else ""
    if errs:
        problems.append(f"{label}: {errs[:2]}")
        errs.clear()
    if BAD.search(txt):
        problems.append(f"{label}: bad text {BAD.search(txt).group(0)!r} in result")
    return txt


def options(pg, sel):
    return [o.get_attribute("value") for o in pg.query_selector_all(f"{sel} option")]


def go(pg):
    pg.click("#f-go")
    pg.wait_for_timeout(120)


def main():
    exe = chrome_path()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()

        # 1. the whole request grid
        ctx, pg, errs = new_page(b)
        equips = options(pg, "#f-equip")
        for eq, kind in itertools.product(equips, ("LI", "SI")):
            pg.select_option("#f-equip", eq)
            pg.click(f'#f-impulse button[data-v="{kind}"]')
            for um in options(pg, "#f-um"):
                pg.select_option("#f-um", um)
                lv = options(pg, "#f-level")
                for level in {lv[0], lv[len(lv) // 2], lv[-1]} if lv else []:
                    pg.select_option("#f-level", level)
                    go(pg)
                    txt = check(f"grid {eq}/{kind}/Um{um}/{level}", pg, errs)
                    if "Cannot run" not in txt and ("Hardware check" not in txt or "OVERLOAD" in txt
                                                    or "over the 90% limit" in txt or "over the rating" in txt):
                        problems.append(f"grid {eq}/{kind}/Um{um}/{level}: hardware check missing or failing")
        print(f"grid done: {checks} requests", flush=True)

        # 2. wrong and extreme measured capacitance
        pg.select_option("#f-equip", equips[0])
        pg.click('#f-impulse button[data-v="LI"]')
        pg.check("#f-measured-chk")
        for v in ("0.09", "-5", "0", "1", "1e9", "99999", "", "90"):
            pg.fill("#f-cpf", v)
            go(pg)
            check(f"capacitance {v!r}", pg, errs)
        pg.uncheck("#f-measured-chk")

        # 3. atmosphere: absurd and edge entries
        pg.check("#f-atmo-chk")
        for t, rh, alt, pr in [("45", "20", "380", ""), ("-60", "50", "", ""), ("80", "200", "", ""),
                               ("20", "-5", "20000", ""), ("20", "50", "", "0"), ("20", "50", "", "200"),
                               ("", "", "", ""), ("abc", "", "", "")]:
            for sel, val in (("#f-temp", t), ("#f-rh", rh), ("#f-alt", alt), ("#f-press", pr)):
                try:
                    pg.fill(sel, val)
                except Exception:
                    pass
            go(pg)
            check(f"atmosphere T={t} RH={rh} alt={alt} P={pr}", pg, errs)
        pg.uncheck("#f-atmo-chk")

        # 4. shot loop with garbage
        go(pg)
        for vals in [("1.2", "50", "1550", "5"), ("0", "0", "0", "0"), ("-1", "-1", "-1", "-1"),
                     ("1e9", "1e9", "1e9", "1e9"), ("", "", "", ""), ("1.2", "50", "1550", "99")]:
            go(pg)                                # a finished loop disables Next setting
            for sel, val in zip(("#m-t1", "#m-t2", "#m-v", "#m-os"), vals):
                if pg.query_selector(sel):
                    pg.fill(sel, val)
            pg.click("#loop-go")
            pg.wait_for_timeout(120)
            check(f"shot loop {vals}", pg, errs, "#loop-out")

        # 5. demo memory, clear memory, hybrid and memory toggles
        for act in ("#mem-demo", "#mem-clear", "#mem-demo"):
            pg.click(act)
            go(pg)
            check(f"memory {act}", pg, errs)
        for chk in ("#f-hybrid", "#f-mem"):
            pg.click(chk)
            go(pg)
            check(f"toggle {chk}", pg, errs)

        # 5b. polarity and series-parallel connections, both impulse kinds
        for kind in ("LI", "SI"):
            pg.click(f'#f-impulse button[data-v="{kind}"]')
            lv = options(pg, "#f-level")
            pg.select_option("#f-level", lv[len(lv) // 2])
            for pol in ("positive", "negative"):
                pg.click(f'#f-polarity button[data-v="{pol}"]')
                go(pg)
                txt = check(f"polarity {kind}/{pol}", pg, errs)
                if pol not in txt:
                    problems.append(f"polarity {kind}/{pol}: not shown in result: {txt[:300]!r}")
                if "Series-parallel stage connection" not in txt:
                    problems.append(f"composition {kind}/{pol}: card missing")
        pg.click('#f-polarity button[data-v="positive"]')

        # 5b2. advanced inputs: typed crest, divider / stray, unit availability
        pg.click('#f-impulse button[data-v="LI"]')
        lv = options(pg, "#f-level")
        pg.select_option("#f-level", lv[len(lv) // 2])
        pg.evaluate("document.querySelector('details.adv').open = true")
        for crest, div, stray in [("1200", "", ""), ("1550", "800", "300"), ("50", "", ""), ("9999", "", ""),
                                  ("-5", "", ""), ("1500", "-1", ""), ("1500", "", "99999"), ("", "5000", "2000"),
                                  ("abc", "", "")]:
            for sel, val in (("#f-crest", crest), ("#f-div", div), ("#f-stray", stray)):
                try:
                    pg.fill(sel, val)
                except Exception:
                    pass
            go(pg)
            txt = check(f"advanced crest={crest!r} div={div!r} stray={stray!r}", pg, errs)
        for sel in ("#f-crest", "#f-div", "#f-stray"):
            pg.fill(sel, "")
        for per in ("1", "2", "4"):
            pg.select_option("#f-per", per)
            go(pg)
            check(f"units per stage {per}", pg, errs)
        pg.select_option("#f-per", "2")
        for v in ("30", "465", "3700"):
            pg.uncheck(f'.av-front[value="{v}"]')
            go(pg)
            check(f"front unit {v} unavailable", pg, errs)
            pg.check(f'.av-front[value="{v}"]')
        for v in ("30", "465", "3700"):
            pg.uncheck(f'.av-front[value="{v}"]')
        go(pg)
        t0 = check("no front units available", pg, errs)
        if "no setting can be built" not in t0:
            problems.append("no front units: missing explanation")
        for v in ("30", "465", "3700"):
            pg.check(f'.av-front[value="{v}"]')
        go(pg)
        t1 = check("waveform and deviation present", pg, errs)
        if "Decision summary" not in t1 or "Deviation" not in t1 or pg.query_selector(".wf svg") is None:
            problems.append("decision summary / deviation / waveform missing")

        # 5b3. lead / loop inductance
        for ind in ("5", "10", "25", "60", "120", "121", "0", "-3", "3", "1e9", "abc", ""):
            try:
                pg.fill("#f-ind", ind)
            except Exception:
                pass
            go(pg)
            txt = check(f"inductance {ind!r}", pg, errs)
            if ind in ("25", "60") and "Lead inductance" not in txt:
                problems.append(f"inductance {ind}: not acknowledged in result")
        for kind in ("SI", "LI"):
            pg.click(f'#f-impulse button[data-v="{kind}"]')
            lv = options(pg, "#f-level")
            pg.select_option("#f-level", lv[len(lv) // 2])
            pg.fill("#f-ind", "40")
            go(pg)
            check(f"inductance 40 {kind}", pg, errs)
        pg.fill("#f-ind", "")

        # 5c. the Physics tab
        pg.click('button[data-view="phys"]')
        pg.wait_for_timeout(150)
        ptxt = check("physics tab", pg, errs, "#v-phys")
        for needle in ("C1 = p", "k(f)", "Kt = k1", "1 + w"):
            if needle not in ptxt:
                problems.append(f"physics tab: missing {needle!r}")
        pg.click('button[data-view="reco"]')

        # 6. every evidence sub-tab
        pg.click('button[data-view="evid"]')
        for btn in pg.query_selector_all("#ev-nav button"):
            btn.click()
            pg.wait_for_timeout(150)
            check(f"evidence tab {btn.inner_text()}", pg, errs, "#v-evid")
        ctx.close()

        # 7. localStorage that throws (private window / blocked site data)
        ctx = b.new_context()
        ctx.add_init_script("""
          const boom = () => { throw new DOMException('blocked', 'SecurityError'); };
          Object.defineProperty(window, 'localStorage', {get: boom});
          Object.defineProperty(window, 'sessionStorage', {get: boom});""")
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.goto(APP)
        pg.wait_for_timeout(400)
        pg.click("#mem-demo")
        go(pg)
        check("localStorage blocked", pg, errs)
        ctx.close()

        # 8. dark mode, phone width
        for scheme, size in (("dark", {"width": 1280, "height": 800}),
                             ("light", {"width": 390, "height": 844})):
            ctx, pg, errs = new_page(b, color_scheme=scheme, viewport=size)
            go(pg)
            check(f"{scheme} {size['width']}px", pg, errs)
            over = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            if over > 4:
                problems.append(f"{scheme} {size['width']}px: page scrolls sideways by {over}px")
            ctx.close()
        b.close()

    print(f"\n{checks} checks, {len(problems)} problem(s)")
    for x in problems:
        print("  -", x)
    return 0 if not problems else 1


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
