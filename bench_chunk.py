"""bench_chunk.py — run the benchmark one hall at a time, serially or in parallel.

Halls are independent: each gets its own `hall(seed, severity)`, its own RNG
seeded from that hall, and its own LabMemory. Nothing crosses between them.
So the benchmark parallelises perfectly by hall, and on a many-core machine
the whole sweep costs about what ONE hall costs.

    python bench_chunk.py --parallel               # all halls at once
    python bench_chunk.py --parallel --halls 5     # 5 halls per severity (P5)
    python bench_chunk.py 3 0 1                    # severity 3, halls 0 and 1
    python bench_chunk.py --part 1 1               # re-run one crashed hall
    python bench_chunk.py --merge                  # merge bench_parts/, summarise
    python bench_chunk.py --summary                # summary from benchmark_results.csv

Why each worker writes its OWN file: the serial path below does a
read-modify-write of one shared CSV. Run that concurrently and the workers
clobber each other. Each parallel worker therefore writes
`bench_parts/sev<S>_hall<H>.csv` and the driver merges afterwards. The part
files are kept, so a crashed hall can be re-run on its own without redoing
the rest.

Why the oracle is not simply early-exited to go faster: `lab.fire` draws from
the noise RNG even when the oracle sets noise to zero, so stopping at the
first feasible setting shifts the random stream and changes every later shot
in that session. Parallelism is the safe speedup; early exit is not.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")
OUT = os.path.join(RESULTS, "benchmark_results.csv")
PARTS = os.path.join(HERE, "bench_parts")


def run_halls(sev, halls, warmup=8, evals=10):
    return B.run([sev], halls_per=len(halls), warmup=warmup, evals=evals,
                 halls=halls)


def _siblings(results_path):
    """Summary/paired/meta paths beside the results file they came from.

    These MUST follow --out rather than being fixed: a smoke run writing to
    a scratch results file otherwise overwrites the real benchmark_summary
    .csv, benchmark_paired.csv and benchmark_meta.json with its throwaway
    numbers. That happened once; hence this function.
    """
    d = os.path.dirname(os.path.abspath(results_path))
    b = os.path.basename(results_path)
    stem = (b[:-len("_results.csv")] if b.endswith("_results.csv")
            else os.path.splitext(b)[0])
    return (os.path.join(d, f"{stem}_summary.csv"),
            os.path.join(d, f"{stem}_paired.csv"),
            os.path.join(d, f"{stem}_meta.json"),
            os.path.join(d, f"{stem}_by_kind.csv"),
            os.path.join(d, f"{stem}_paired_by_kind.csv"))


def write_summary(df, results_path=OUT):
    """Summary, paired tests and meta from a results frame. Shared by every
    path so the serial and parallel runs report identically."""
    summary_path, paired_path, meta_path, kind_path, pkind_path = \
        _siblings(results_path)
    summ, inf = B.summarise(df)
    summ.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"infeasible": int(inf), "total": int(len(df))}, fh)
    print(summ.to_string(index=False))
    print("infeasible", inf, "of", len(df))
    pt = B.all_paired_tests(df)
    pt.to_csv(paired_path, index=False)
    print("\nPAIRED TESTS — first block is the headline: vs current "
          "practice, severities 3-4, shots")
    print(pt.to_string(index=False))
    B.summarise_by_kind(df).to_csv(kind_path, index=False)
    pk = B.paired_by_kind(df)
    pk.to_csv(pkind_path, index=False)
    print("\nBY IMPULSE TYPE (P4): exploratory, outside the Bonferroni family")
    print(pk[pk.strategy.isin(["lab_memory", "app_memory", "app_loop",
                               "calibration"])].to_string(index=False))
    return summ


def part_path(sev, hall):
    return os.path.join(PARTS, f"sev{sev}_hall{hall}.csv")


def _one_hall(sev, hall, warmup, evals):
    """Launch one hall as its own process. Returns (sev, hall, ok, stderr)."""
    out = part_path(sev, hall)
    cmd = [sys.executable, os.path.abspath(__file__), "--part", str(sev),
           str(hall), "--warmup", str(warmup), "--evals", str(evals)]
    env = dict(os.environ)
    # The linear algebra here is 4x4 — BLAS threads buy nothing and would
    # oversubscribe badly with one pool per worker.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    return sev, hall, r.returncode == 0 and os.path.exists(out), r.stderr


def merge_parts(paths, out=OUT):
    """Concatenate hall part files into one results CSV."""
    frames = [pd.read_csv(p) for p in paths
              if os.path.exists(p) and os.path.getsize(p) > 0]
    if not frames:
        sys.exit("no hall produced results — nothing to merge.")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nmerged {len(frames)} halls -> {os.path.basename(out)} "
          f"({len(df)} sessions)")
    return df


def parallel(severities, n_halls, warmup, evals, jobs, out=OUT):
    os.makedirs(PARTS, exist_ok=True)
    tasks = [(s, h) for s in severities for h in range(n_halls)]
    jobs = jobs or min(len(tasks), os.cpu_count() or 4)
    print(f"{len(tasks)} halls on {jobs} workers "
          f"({n_halls} per severity, {evals} scored sessions each)\n"
          f"Expect roughly the cost of ONE hall if the workers all fit.",
          flush=True)

    failed, done = [], 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_one_hall, s, h, warmup, evals): (s, h)
                for s, h in tasks}
        for f in as_completed(futs):
            sev, hall, ok, err = f.result()
            done += 1
            if ok:
                print(f"  [{done}/{len(tasks)}] severity {sev} hall {hall} ok",
                      flush=True)
            else:
                failed.append((sev, hall))
                tail = (err or "").strip().splitlines()[-1:] or ["no stderr"]
                print(f"  [{done}/{len(tasks)}] severity {sev} hall {hall} "
                      f"FAILED: {tail[0]}", flush=True)

    df = merge_parts([part_path(s, h) for s, h in tasks], out)
    if failed:
        # Never summarise over a partial sweep without saying so: a missing
        # hall silently changes every severity mean it contributed to.
        print(f"WARNING: {len(failed)} hall(s) failed and are NOT in the "
              f"merge: {failed}")
        print("Re-run just those, then merge:  " + "  ".join(
            f"python bench_chunk.py --part {s} {h}" for s, h in failed)
            + "  python bench_chunk.py --merge")
    print()
    write_summary(df, out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pos", nargs="*", help="severity then hall numbers")
    ap.add_argument("--summary", action="store_true",
                    help="rebuild summary from benchmark_results.csv")
    ap.add_argument("--parallel", action="store_true",
                    help="run every hall concurrently, then merge")
    ap.add_argument("--merge", action="store_true",
                    help="merge every part in bench_parts/, then summarise")
    ap.add_argument("--part", nargs=2, type=int, metavar=("SEV", "HALL"),
                    help="worker mode: one hall to its own part file")
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--halls", type=int, default=3,
                    help="halls per severity (default 3, as benchmark.py)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.part:
        sev, hall = a.part
        os.makedirs(PARTS, exist_ok=True)
        run_halls(sev, [hall], warmup=a.warmup, evals=a.evals).to_csv(
            part_path(sev, hall), index=False)
        sys.exit()

    if a.merge:
        import glob
        parts = sorted(glob.glob(os.path.join(PARTS, "sev*_hall*.csv")))
        write_summary(merge_parts(parts, a.out), a.out)
        sys.exit()

    if a.summary:
        # no drop_duplicates: two distinct sessions in one hall can have
        # identical outcomes, and deduplicating would silently delete them.
        # Re-run chunks are already replaced by (severity, hall) on write.
        write_summary(pd.read_csv(a.out), a.out)
        sys.exit()

    if a.parallel:
        parallel(a.severities, a.halls, a.warmup, a.evals, a.jobs, a.out)
        sys.exit()

    if not a.pos:
        ap.error("give a severity and hall numbers, or --parallel / --summary")
    sev = int(a.pos[0])
    halls = [int(x) for x in a.pos[1:]]
    new = run_halls(sev, halls)
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        old = pd.read_csv(a.out)
        old = old[~((old.severity == sev) & old.hall.isin(new.hall))]
        new = pd.concat([old, new])
    new.to_csv(a.out, index=False)
    print(f"severity {sev} halls {halls}: {len(new)} rows total")
