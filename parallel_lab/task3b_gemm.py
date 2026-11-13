"""Phase 3B: parallel GEMM variants and tile-size sweep.

Usage:
    python task3b_gemm.py <StudentID> [--pphys 6] [--l1 32] [--l2 512]

Run task3a_roofline.py first (it writes roofline.json, used for the roofline
mapping). --n is for quick debugging only; VERIFY_HASH is valid only at N=384.
"""
import argparse
import json
import multiprocessing as mp
import random
from array import array
from multiprocessing import shared_memory
from pathlib import Path

from harness import run_benchmark, emit_verify_tag

N_DEFAULT = 384
BS_LIST = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
WARMUP, TIMED = 2, 7
I_NAIVE = 0.125  # FLOP/Byte, given in the task

_N = 0
_A = None
_B = None
_C = None
_KEEP = []


def init_worker(name_a, name_b, name_c, n):
    global _N, _A, _B, _C
    _N = n
    nbytes = n * n * 8
    views = []
    for name in (name_a, name_b, name_c):
        shm = shared_memory.SharedMemory(name=name)
        raw = shm.buf[:nbytes]
        _KEEP.extend([shm, raw])
        views.append(raw.cast("d"))
    _A, _B, _C = views


def v1_rows(r0, r1):
    """Naive i-j-k: inner loop walks B with stride N."""
    n, A, B, C = _N, _A, _B, _C
    for i in range(r0, r1):
        for j in range(n):
            dot = 0.0
            for k in range(n):
                dot += A[i * n + k] * B[k * n + j]
            C[i * n + j] = dot


def v2_rows(r0, r1):
    """Loop-reordered i-k-j: a_ik hoisted, inner loop over contiguous row slices."""
    n, A, B, C = _N, _A, _B, _C
    for i in range(r0, r1):
        ai = i * n
        crow = [0.0] * n
        for k in range(n):
            a_ik = A[ai + k]
            kb = k * n
            brow = B[kb:kb + n]
            crow = [c + a_ik * b for c, b in zip(crow, brow)]
        C[ai:ai + n] = array("d", crow)


def v3_rows(r0, r1, bs):
    """6-loop cache-blocked GEMM. Micro-kernel order (i, k, j), a_ik hoisted,
    B tile rows hoisted out of the i-loop and reused for every row of the tile."""
    n, A, B, C = _N, _A, _B, _C
    for ii in range(r0, r1, bs):
        i_end = min(ii + bs, r1)
        for jj in range(0, n, bs):
            j_end = min(jj + bs, n)
            ctile = [[0.0] * (j_end - jj) for _ in range(i_end - ii)]
            for kk in range(0, n, bs):
                k_end = min(kk + bs, n)
                btile = [list(B[k * n + jj:k * n + j_end]) for k in range(kk, k_end)]
                for i in range(ii, i_end):
                    ai = i * n
                    crow = ctile[i - ii]
                    for k in range(kk, k_end):
                        a_ik = A[ai + k]
                        crow = [c + a_ik * b for c, b in zip(crow, btile[k - kk])]
                    ctile[i - ii] = crow
            for i in range(ii, i_end):
                C[i * n + jj:i * n + j_end] = array("d", ctile[i - ii])


def gemm_task(args):
    variant, r0, r1, bs = args
    if variant == 1:
        v1_rows(r0, r1)
    elif variant == 2:
        v2_rows(r0, r1)
    else:
        v3_rows(r0, r1, bs)


def split(n, p):
    base, extra = divmod(n, p)
    out, start = [], 0
    for t in range(p):
        size = base + (1 if t < extra else 0)
        out.append((start, start + size))
        start += size
    return out


def fits_in(nbytes, l1_kb, l2_kb):
    if nbytes <= l1_kb * 1024:
        return "L1d"
    if nbytes <= l2_kb * 1024:
        return "L2"
    return "L3"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id", type=int)
    ap.add_argument("--pphys", type=int, default=6)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--l1", type=int, default=32, help="L1d per core, KB")
    ap.add_argument("--l2", type=int, default=512, help="L2 per core, KB")
    ap.add_argument("--roofline", default=None)
    args = ap.parse_args()

    n, p = args.n, args.pphys
    if n != N_DEFAULT:
        print("WARNING: --n changed, VERIFY_HASH is NOT valid for submission")
    nbytes = n * n * 8
    flops = 2 * n ** 3
    rows = split(n, p)
    print(f"N={n}, P_phys={p}, rows per worker={[r1 - r0 for r0, r1 in rows]}, "
          f"footprint={3 * nbytes / 1e6:.2f} MB")

    shms = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(3)]
    raws = [s.buf[:nbytes] for s in shms]
    views = [r.cast("d") for r in raws]
    A, B, C = views
    try:
        rng = random.Random(2026)
        A[:] = array("d", [rng.random() for _ in range(n * n)])
        B[:] = array("d", [rng.random() for _ in range(n * n)])

        def gflops(st):
            return flops / (st.median_ms / 1e3) / 1e9

        def maxdiff(ref):
            return max(abs(x - y) for x, y in zip(C, ref))

        results = []  # (label, bs, stats, gflops, diff)
        with mp.Pool(processes=p, initializer=init_worker,
                     initargs=(shms[0].name, shms[1].name, shms[2].name, n)) as pool:

            def bench(variant, bs):
                tasks = [(variant, r0, r1, bs) for r0, r1 in rows]
                return run_benchmark(lambda: pool.map(gemm_task, tasks, chunksize=1),
                                     warmup_runs=WARMUP, timed_runs=TIMED)

            # Variant 2 first: it is the numerical reference
            st2 = bench(2, 0)
            ref = array("d", C)
            # sanity check of the reference against direct dot products
            for (i, j) in [(0, 0), (n - 1, n - 1), (n // 2, n // 3), (3, n - 2), (n - 4, 5)]:
                direct = sum(A[i * n + k] * B[k * n + j] for k in range(n))
                assert abs(direct - ref[i * n + j]) < 1e-9, "reference GEMM is wrong"
            print(f"  V2: median={st2.median_ms:.1f} ms", flush=True)

            st1 = bench(1, 0)
            d1 = maxdiff(ref)
            print(f"  V1: median={st1.median_ms:.1f} ms", flush=True)
            results.append(("V1 naive i-j-k", None, st1, gflops(st1), d1))
            results.append(("V2 i-k-j", None, st2, gflops(st2), 0.0))

            for bs in BS_LIST:
                st = bench(3, bs)
                d = maxdiff(ref)
                results.append(("V3 blocked", bs, st, gflops(st), d))
                print(f"  V3 BS={bs}: median={st.median_ms:.1f} ms", flush=True)

        # ---------- Table ----------
        print("\nTable 3 (GEMM)")
        print("| Variant | BS | W_tile (KB) | Fits in | Median (ms) | IQR (ms) | GFLOP/s | max diff vs V2 |")
        print("|---|---|---|---|---|---|---|---|")
        for label, bs, st, g, d in results:
            if bs is None:
                w, fit = "-", "-"
                bs_s = "-"
            else:
                wb = 3 * bs * bs * 8
                w, fit, bs_s = f"{wb / 1024:.1f}", fits_in(wb, args.l1, args.l2), str(bs)
            print(f"| {label} | {bs_s} | {w} | {fit} | {st.median_ms:.1f} | "
                  f"{st.iqr_ms:.1f} | {g:.4f} | {d:.2e} |")
        worst = max(r[4] for r in results)
        print(f"\nMax |C - C(V2)| over all variants: {worst:.2e} "
              f"({'OK' if worst < 1e-9 else 'FAIL'}, threshold 1e-9)")

        v1 = results[0]
        v2 = results[1]
        v3_all = [r for r in results if r[1] is not None]
        best = max(v3_all, key=lambda r: r[3])
        bs_star = best[1]
        print(f"BS* = {bs_star} (GFLOP/s = {best[3]:.4f}), "
              f"W_tile = {3 * bs_star ** 2 * 8 / 1024:.1f} KB, "
              f"fits in {fits_in(3 * bs_star ** 2 * 8, args.l1, args.l2)} "
              f"(L1d={args.l1} KB, L2={args.l2} KB)")
        if bs_star > rows[0][1] - rows[0][0]:
            print(f"Note: each worker owns {rows[0][1] - rows[0][0]} rows, so the "
                  f"tile height is clipped to that for BS > {rows[0][1] - rows[0][0]}.")

        # ---------- Roofline mapping ----------
        rl_path = Path(args.roofline) if args.roofline else Path(__file__).resolve().with_name("roofline.json")
        if rl_path.exists():
            rl = json.loads(rl_path.read_text())
            bw, peak = rl["bw_gbs"], rl["p_gflops"]
            print(f"\nRoofline (from {rl_path.name}): BW={bw:.2f} GB/s, "
                  f"P_sustained={peak:.4f} GFLOP/s, I*={peak / bw:.5f} FLOP/B")
            print("| Variant | I (FLOP/B) | Attainable (GFLOP/s) | Achieved (GFLOP/s) | % of attainable | Bound |")
            print("|---|---|---|---|---|---|")
            rows_rl = [
                ("V1 naive", I_NAIVE, v1[3]),
                ("V2 i-k-j (I assumed = naive)", I_NAIVE, v2[3]),
                (f"V3 BS*={bs_star}", bs_star / 12.0, best[3]),
            ]
            for name, inten, ach in rows_rl:
                att = min(peak, bw * inten)
                bound = "compute" if bw * inten >= peak else "memory"
                print(f"| {name} | {inten:.3f} | {att:.4f} | {ach:.4f} | "
                      f"{100 * ach / att:.1f}% | {bound} |")
            print("(a value above 100% means the measured ceiling in task3a is not "
                  "a true upper bound for that kernel)")
        else:
            print("\nroofline.json not found: run task3a_roofline.py first, "
                  "or pass --roofline <path>")

        print()
        print(emit_verify_tag("PHASE3", args.student_id, v1[3], best[3]))
    finally:
        for v in views:
            v.release()
        for r in raws:
            r.release()
        for s in shms:
            s.close()
            s.unlink()
