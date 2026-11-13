"""Phase 3A: empirical roofline ceilings (sustained bandwidth, compute, ridge point).

Usage:
    python task3a_roofline.py [--pphys 6]

--mb and --fma are for quick debugging only; use the defaults for the report.
Results are saved to roofline.json (next to this script) for task3b_gemm.py.
"""
import argparse
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

from harness import run_benchmark

MB_DEFAULT = 96            # size of each buffer, in 10^6 bytes
FMA_FLOPS_DEFAULT = 8.0e7  # FLOPs per worker (each x = x*a + b counts as 2)

_KEEP = []
_MV_A = None
_MV_B = None


def init_copy(name_a: str, name_b: str, nbytes: int) -> None:
    global _MV_A, _MV_B
    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    _KEEP.extend([shm_a, shm_b])
    _MV_A = shm_a.buf[:nbytes]
    _MV_B = shm_b.buf[:nbytes]


def copy_slice(bounds) -> None:
    s, e = bounds
    _MV_A[s:e] = _MV_B[s:e]  # native memcpy


def fma_kernel(iters: int) -> float:
    a = 0.9999999
    b = 1e-7
    x0 = 0.10
    x1 = 0.11
    x2 = 0.12
    x3 = 0.13
    x4 = 0.14
    x5 = 0.15
    x6 = 0.16
    x7 = 0.17
    x8 = 0.18
    x9 = 0.19
    x10 = 0.20
    x11 = 0.21
    x12 = 0.22
    x13 = 0.23
    x14 = 0.24
    x15 = 0.25
    for _ in range(iters):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b
    return (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
            + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15)


def split(n: int, p: int):
    base, extra = divmod(n, p)
    out, start = [], 0
    for t in range(p):
        size = base + (1 if t < extra else 0)
        out.append((start, start + size))
        start += size
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pphys", type=int, default=6)
    ap.add_argument("--mb", type=int, default=MB_DEFAULT)
    ap.add_argument("--fma", type=float, default=FMA_FLOPS_DEFAULT)
    args = ap.parse_args()

    p = args.pphys
    nbytes = args.mb * 10**6
    if args.mb != MB_DEFAULT or args.fma != FMA_FLOPS_DEFAULT:
        print("WARNING: defaults changed, results are NOT valid for submission")
    print(f"P_phys={p}, buffer size={args.mb} MB each ({2 * args.mb} MB total)")

    # ---------- 1. Sustained memory bandwidth ----------
    shm_a = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        fill = b"\x5a" * nbytes
        shm_a.buf[:nbytes] = fill   # commit all pages before timing
        shm_b.buf[:nbytes] = fill
        del fill

        chunks = split(nbytes, p)
        with mp.Pool(processes=p, initializer=init_copy,
                     initargs=(shm_a.name, shm_b.name, nbytes)) as pool:
            st = run_benchmark(lambda: pool.map(copy_slice, chunks, chunksize=1),
                               warmup_runs=3, timed_runs=9)
    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()

    t_s = st.median_ms / 1e3
    bw = (2 * nbytes) / t_s / 1e9   # 1 read + 1 write per copied byte
    print("\nPart 1: sustained bandwidth")
    print(f"  copy time: median={st.median_ms:.2f} ms, IQR={st.iqr_ms:.2f} ms, "
          f"min={st.min_ms:.2f}, max={st.max_ms:.2f}")
    print(f"  BW_sustained = 2 x {nbytes:,} B / {t_s:.4f} s = {bw:.2f} GB/s")

    # ---------- 2. Sustained Python compute ceiling ----------
    iters = int(args.fma / 32)  # 16 lines x 2 FLOPs per loop iteration
    total_flops = p * iters * 32
    with mp.Pool(processes=p) as pool:
        st2 = run_benchmark(lambda: pool.map(fma_kernel, [iters] * p, chunksize=1),
                            warmup_runs=2, timed_runs=7)
    t2_s = st2.median_ms / 1e3
    p_sust = total_flops / t2_s / 1e9
    print("\nPart 2: sustained Python compute")
    print(f"  {iters:,} iterations x 32 FLOPs = {iters * 32:,} FLOPs per worker, "
          f"{total_flops:,} total")
    print(f"  time: median={st2.median_ms:.1f} ms, IQR={st2.iqr_ms:.1f} ms")
    print(f"  P_sustained = {p_sust:.4f} GFLOP/s "
          f"({p_sust / p:.4f} per worker)")

    # ---------- 3. Ridge point ----------
    ridge = p_sust / bw
    print("\nPart 3: ridge point")
    print(f"  I* = P_sustained / BW_sustained = {p_sust:.4f} / {bw:.2f} "
          f"= {ridge:.5f} FLOP/Byte")

    out = Path(__file__).resolve().with_name("roofline.json")
    out.write_text(json.dumps({
        "p_phys": p,
        "bw_gbs": bw,
        "p_gflops": p_sust,
        "ridge": ridge,
    }, indent=2))
    print(f"\nSaved {out.name} (used by task3b_gemm.py)")
