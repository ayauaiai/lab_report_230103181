"""Phase 2: strong scaling (Amdahl) vs weak scaling (Gustafson).

Usage:
    python task2_scaling.py <StudentID> [--pphys 6]

--scale is for quick debugging only (shrinks N and N0); the
VERIFY_HASH is valid only with the default --scale 1.0.
"""
import argparse
import math
import multiprocessing as mp
import statistics
import time

from harness import run_benchmark, emit_verify_tag

N_STRONG = 1_600_000   # 1.6 x 10^6
N0_WEAK = 250_000      # 2.5 x 10^5
STAGE_A_FRAC = 0.05
WARMUP, TIMED = 2, 7


def stage_a(n: int) -> float:
    """Serial stage: first 5% of elements, 12 transcendental ops each
    (6 x sin + 6 x sqrt), with a data dependence between elements."""
    m = int(STAGE_A_FRAC * n)
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(m):
        x = 0.001 * (i + 1) + 1e-9 * acc
        for _ in range(6):
            x = sqrt(sin(x) + 2.0)
        acc += x
    return acc


def kernel(bounds):
    """Parallel stage: 35 multiply-add + sin iterations per element, no shared writes."""
    start, end = bounds
    sin = math.sin
    total = 0.0
    for i in range(start, end):
        x = (i % 1000) * 1e-3 + 0.1
        for _ in range(35):
            x = x * 0.999 + 0.001 + 0.5 * sin(x)
        total += x
    return total


def split(n: int, p: int):
    base, extra = divmod(n, p)
    out, start = [], 0
    for t in range(p):
        size = base + (1 if t < extra else 0)
        out.append((start, start + size))
        start += size
    return out


def run_point(p: int, n: int):
    """Returns (BenchStats, median Stage A time in ms) for p workers, N = n."""
    chunks = split(n, p)
    a_log = []
    with mp.Pool(processes=p) as pool:  # created OUTSIDE the timed region

        def trial():
            t0 = time.perf_counter_ns()
            stage_a(n)
            t1 = time.perf_counter_ns()
            pool.map(kernel, chunks, chunksize=1)
            a_log.append((t1 - t0) / 1e6)

        st = run_benchmark(trial, warmup_runs=WARMUP, timed_runs=TIMED)
    a_med = statistics.median(a_log[WARMUP:])  # timed runs only
    return st, a_med


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id", type=int)
    ap.add_argument("--pphys", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    p_phys = args.pphys
    p_log = mp.cpu_count()
    n_strong = int(N_STRONG * args.scale)
    n0 = int(N0_WEAK * args.scale)
    if args.scale != 1.0:
        print("WARNING: --scale changed, VERIFY_HASH is NOT valid for submission")
    print(f"P_phys={p_phys}, P_log={p_log}, N_strong={n_strong}, N0={n0}")

    ps = list(range(1, p_log + 1))

    # ---------- Experiment 2A: strong scaling ----------
    print("\nExperiment 2A: strong scaling")
    t_strong, a_strong = {}, {}
    for p in ps:
        st, a_med = run_point(p, n_strong)
        t_strong[p], a_strong[p] = st.median_ms, a_med
        print(f"  p={p}: median={st.median_ms:.1f} ms (Stage A {a_med:.1f} ms)", flush=True)

    t1 = t_strong[1]
    s_ser = a_strong[1] / t1
    s_inf = 1.0 / s_ser
    s_strong = {p: t1 / t_strong[p] for p in ps}

    def amdahl(p):
        return 1.0 / (s_ser + (1.0 - s_ser) / p)

    print("\nTable 2A (Strong Scaling)")
    print("| p | T_strong (ms) | S_strong | E_strong | Amdahl S(p) |")
    print("|---|---|---|---|---|")
    for p in ps:
        print(f"| {p} | {t_strong[p]:.1f} | {s_strong[p]:.3f} | "
              f"{s_strong[p] / p:.3f} | {amdahl(p):.3f} |")

    print(f"\nSerial fraction s = T_A(1)/T_strong(1) = {a_strong[1]:.1f} / {t1:.1f} = {s_ser:.4f}")
    print(f"Amdahl S_inf = 1/s = {s_inf:.2f}")
    print(f"Amdahl predicted S({p_log}) = {amdahl(p_log):.3f}")
    print(f"Observed S_strong({p_log}) = {s_strong[p_log]:.3f}")

    # ---------- Marginal speedup / SMT boundary ----------
    print("\nMarginal speedup per added worker, m(p) = S(p) - S(p-1)")
    print("| p | m(p) |")
    print("|---|---|")
    marg = {p: s_strong[p] - s_strong[p - 1] for p in ps if p >= 2}
    for p in sorted(marg):
        print(f"| {p} | {marg[p]:.3f} |")

    if p_phys >= 2 and p_log > p_phys:
        m_before, m_after = marg[p_phys], marg[p_phys + 1]
        drop_step = (1.0 - m_after / m_before) * 100.0
        avg_phys = (s_strong[p_phys] - s_strong[1]) / (p_phys - 1)
        avg_smt = (s_strong[p_log] - s_strong[p_phys]) / (p_log - p_phys)
        drop_avg = (1.0 - avg_smt / avg_phys) * 100.0
        print(f"\nBoundary p={p_phys} -> {p_phys + 1}: m({p_phys})={m_before:.3f}, "
              f"m({p_phys + 1})={m_after:.3f}, drop = {drop_step:.1f}%")
        print(f"Avg marginal speedup, p=2..{p_phys}: {avg_phys:.3f}; "
              f"p={p_phys + 1}..{p_log}: {avg_smt:.3f}; drop = {drop_avg:.1f}%")

    # ---------- Experiment 2B: weak scaling ----------
    print("\nExperiment 2B: weak scaling")
    t_weak, a_weak = {}, {}
    for p in ps:
        st, a_med = run_point(p, p * n0)
        t_weak[p], a_weak[p] = st.median_ms, a_med
        print(f"  p={p}: median={st.median_ms:.1f} ms (Stage A {a_med:.1f} ms)", flush=True)

    print("\nTable 2B (Weak Scaling)")
    print("| p | N(p) | T_weak (ms) | Stage A (ms) | E_weak | alpha | S_Gustafson |")
    print("|---|---|---|---|---|---|---|")
    s_gus = {}
    for p in ps:
        alpha = a_weak[p] / t_weak[p]
        s_gus[p] = p - alpha * (p - 1)
        print(f"| {p} | {p * n0} | {t_weak[p]:.1f} | {a_weak[p]:.1f} | "
              f"{t_weak[1] / t_weak[p]:.3f} | {alpha:.4f} | {s_gus[p]:.3f} |")

    print()
    print(emit_verify_tag("PHASE2", args.student_id, s_strong[p_log], s_gus[p_log]))
