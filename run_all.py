"""
run_all.py — verify the physics, then build the dashboard.

    python run_all.py              # checks + dashboard from existing results
    python run_all.py --benchmark  # also re-run the full benchmark, in parallel

The benchmark is slow because every session runs six strategies against a
hidden lab and an exhaustive oracle, so it is not re-run by default. Its
results (benchmark_results.csv) are committed alongside the code.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def step(label, args):
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}", flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\nFAILED: {' '.join(args)} — later steps depend on it.")


if __name__ == "__main__":
    step("[1] Circuit physics against Kuffel", ["circuit.py"])
    step("[2] IEC 60060-1 waveform evaluation", ["waveform.py"])
    step("[2b] Operator input handling and IEC 60071 levels",
         ["recommend.py", "--self-check"])
    step("[2c] CPRI's own dataset: their kNN reproduced, then beaten or not",
         ["regression.py", "--cpri"])
    have = os.path.exists(os.path.join(RESULTS, "benchmark_summary.csv"))
    if "--benchmark" in sys.argv or not have:
        step("[3] Benchmark: six strategies across the severity sweep",
             ["bench_chunk.py", "--parallel"])
    else:
        print("\n[3] Using existing benchmark results "
              "(pass --benchmark to re-run in parallel)")
    if not os.path.exists(os.path.join(RESULTS, "regression_results.csv")):
        step("[3b] Regression benchmark vs CPRI weighted KNN",
             ["regression.py", "--n", "360"])
        step("[3c] Learning curve", ["regression.py", "--reuse", "--curve"])
    else:
        print("[3b] Using existing regression results "
              "(python regression.py --n 360 to rebuild)")
    step("[4] Dashboard", ["dashboard.py"])
    print("\nDone. Open impulse_dashboard.html in a browser.")
