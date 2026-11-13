"""Phase 4: skewed workload, load imbalance and dynamic chunk scheduling.

Usage:
    python task4_scheduling.py <StudentID> [--pphys 6]

--n, --warmup and --timed are for tuning/debugging; VERIFY_HASH is valid
only for the lab's default N = 12000.
"""
import argparse
import math
import multiprocessing as mp
import statistics
import time

from harness import run_benchmark, emit_verify_tag

N_DEFAULT = 12_000
CHUNKS = [1, 4, 16, 64, 256, 1024]


def item(i: int) -> float:
    """Item i performs K(i) = 1 + floor((i/60)^2) trigonometric iterations."""
    k = 1 + (i * i) // 3600
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    x = 0.5 + i * 1e-4
    for _ in range(k):
        x = sqrt(abs(sin(x) + cos(x))) + 0.1
    return x


def worker_block(t, p, n, times):
    t0 = time.perf_counter_ns()
    acc = 0.0
    for i in range(t * n // p, (t + 1) * n // p):
        acc += item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def worker_cyclic(t, p, n, times):
    t0 = time.perf_counter_ns()
    acc = 0.0
    for i in range(t, n, p):
        acc += item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def worker_dynamic(t, n, c, counter, times):
    t0 = time.perf_counter_ns()
    acc = 0.0
    while True:
        with counter.get_lock():
            start = counter.value
            counter.value = start + c
        if start >= n:
            break
        for i in range(start, min(start + c, n)):
            acc += item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def make_trial(strategy, c, p, n, times, counter, log):
    def trial():
        for t in range(p):
            times[t] = 0.0
        counter.value = 0
        if strategy == "block":
            procs = [mp.Process(target=worker_block, args=(t, p, n, times)) for t in range(p)]
        elif strategy == "cyclic":
            procs = [mp.Process(target=worker_cyclic, args=(t, p, n, times)) for t in range(p)]
        else:
            procs = [mp.Process(target=worker_dynamic, args=(t, n, c, counter, times))
                     for t in range(p)]
        t0 = time.perf_counter_ns()
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        twall = (time.perf_counter_ns() - t0) / 1e6
        log.append((twall, times[:]))
    return trial


def measure(strategy, c, p, n, times, counter, warmup, timed):
    log = []
    st = run_benchmark(make_trial(strategy, c, p, n, times, counter, log),
                       warmup_runs=warmup, timed_runs=timed)
    runs = sorted(log[warmup:], key=lambda r: r[0])
    twall_mid, ts = runs[len(runs) // 2]  # the run with the median wall time
    mean_t = statistics.fmean(ts)
    lam = max(ts) / mean_t
    util = sum(ts) / (p * twall_mid)
    return {
        "strategy": strategy, "c": c, "median": st.median_ms, "iqr": st.iqr_ms,
        "lam": lam, "max_t": max(ts), "mean_t": mean_t, "util": util, "ts": ts,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id", type=int)
    ap.add_argument("--pphys", type=int, default=6)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timed", type=int, default=5)
    args = ap.parse_args()

    p, n = args.pphys, args.n
    if n != N_DEFAULT:
        print("WARNING: --n changed, VERIFY_HASH is NOT valid for submission")
    total_k = sum(1 + (i * i) // 3600 for i in range(n))
    print(f"N={n}, P_phys={p}, K(0)=1, K({n - 1})={1 + ((n - 1) ** 2) // 3600}, "
          f"total iterations={total_k:,}")
    print(f"Runs per configuration: {args.warmup} warm-up + {args.timed} timed\n")

    times = mp.Array("d", p)
    counter = mp.Value("i", 0)

    results = []
    plan = [("block", 0), ("cyclic", 0)] + [("dynamic", c) for c in CHUNKS]
    for strategy, c in plan:
        r = measure(strategy, c, p, n, times, counter, args.warmup, args.timed)
        results.append(r)
        label = strategy if strategy != "dynamic" else f"dynamic C={c}"
        print(f"  {label}: Twall={r['median']:.1f} ms, lambda={r['lam']:.4f}", flush=True)

    names = {"block": "S1 contiguous block", "cyclic": "S2 cyclic", "dynamic": "S3 dynamic"}
    print("\nTable 4 (Scheduling and load imbalance)")
    print("| Strategy | C | T_wall (ms) | IQR (ms) | lambda | max t_worker (ms) | mean t_worker (ms) | Utilization |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        c_s = str(r["c"]) if r["strategy"] == "dynamic" else "-"
        print(f"| {names[r['strategy']]} | {c_s} | {r['median']:.1f} | {r['iqr']:.1f} | "
              f"{r['lam']:.4f} | {r['max_t']:.1f} | {r['mean_t']:.1f} | {r['util']:.3f} |")

    dyn = [r for r in results if r["strategy"] == "dynamic"]
    best = min(dyn, key=lambda r: r["median"])
    blk, cyc = results[0], results[1]
    print(f"\nBest dynamic chunk size C* = {best['c']} "
          f"(T_wall = {best['median']:.1f} ms, lambda = {best['lam']:.4f})")
    print(f"Speedup of C* over block static: {blk['median'] / best['median']:.2f}x; "
          f"over cyclic: {cyc['median'] / best['median']:.2f}x")

    print("\nPer-worker active time t_worker[t] (ms), run with median T_wall:")
    for r in (blk, cyc, best):
        tag = names[r["strategy"]] + (f" C={r['c']}" if r["strategy"] == "dynamic" else "")
        print(f"  {tag}: " + ", ".join(f"{x:.1f}" for x in r["ts"]))

    print("\nRAW values (4 decimals):")
    print(f"  lambda_block={blk['lam']:.4f}  lambda_cyclic={cyc['lam']:.4f}  "
          f"lambda_best_dynamic={best['lam']:.4f}")
    print(f"  Twall_block={blk['median']:.4f}  Twall_cyclic={cyc['median']:.4f}  "
          f"Twall_best_dynamic={best['median']:.4f}")

    print("\nASSUMPTION: lambda = max(t_worker) / mean(t_worker); the hash below uses "
          "(lambda_block, lambda_best_dynamic). Check the lab text for the exact definition.")
    print(emit_verify_tag("PHASE4", args.student_id, blk["lam"], best["lam"]))
