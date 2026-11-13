"""Phase 1: false sharing diagnosis with Karp-Flatt analysis.

Usage:
    python task1_falsesharing.py <StudentID> [--pphys 6]

--iters is for quick debugging only; the VERIFY_HASH is valid only
with the default M = 2.4e7.
"""
import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag

M_TOTAL = 24_000_000  # 2.4 x 10^7 total steps
SHM_SIZE = 4096


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc  # direct store to the shared cache line
    del arr  # release buffer export before closing
    shm.close()


def run_once(shm: shared_memory.SharedMemory, p: int, stride: int, m_total: int) -> None:
    shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)  # reset so every run is deterministic
    iters = m_total // p
    procs = [
        mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
        for t in range(p)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def bench(shm, p, stride, m_total):
    return run_benchmark(
        lambda: run_once(shm, p, stride, m_total),
        warmup_runs=2,
        timed_runs=7,
    )


def scaling_values(p_log: int):
    ps = []
    p = 1
    while p <= p_log:
        ps.append(p)
        p *= 2
    if ps[-1] != p_log:
        ps.append(p_log)  # include P_log itself (e.g. 12)
    return ps


def karp_flatt(sp: float, p: int) -> float:
    return (1.0 / sp - 1.0 / p) / (1.0 - 1.0 / p)


def scaling_table(shm, ps, stride, m_total):
    rows = []
    t1 = None
    for p in ps:
        st = bench(shm, p, stride, m_total)
        if p == 1:
            t1 = st.median_ms
        sp = t1 / st.median_ms
        ep = sp / p
        e = karp_flatt(sp, p) if p >= 2 else None
        rows.append((p, st.median_ms, st.iqr_ms, sp, ep, e))
        print(f"  stride={stride} p={p}: median={st.median_ms:.1f} ms", flush=True)
    return rows


def print_scaling(title, rows):
    print(f"\n{title}")
    print("| p | Median (ms) | IQR (ms) | Sp | Ep | e(p) |")
    print("|---|---|---|---|---|---|")
    for p, med, iqr, sp, ep, e in rows:
        e_s = "n/a" if e is None else f"{e:.4f}"
        print(f"| {p} | {med:.1f} | {iqr:.1f} | {sp:.3f} | {ep:.3f} | {e_s} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id", type=int)
    ap.add_argument("--pphys", type=int, default=6)
    ap.add_argument("--iters", type=int, default=M_TOTAL)
    args = ap.parse_args()

    p_phys = args.pphys
    p_log = mp.cpu_count()
    m_total = args.iters
    if m_total != M_TOTAL:
        print("WARNING: --iters changed, VERIFY_HASH is NOT valid for submission")
    print(f"P_phys={p_phys}, P_log={p_log}, M={m_total}")

    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
    try:
        # Table 1A: stride sweep at p = P_phys
        print(f"\nStride sweep at p={p_phys}")
        strides = [1, 2, 4, 8, 16, 32]
        t1a = []
        for s in strides:
            st = bench(shm, p_phys, s, m_total)
            mops = (m_total / 1e6) / (st.median_ms / 1e3)
            t1a.append((s, s * 8, st.median_ms, st.iqr_ms, mops))
            print(f"  stride={s}: median={st.median_ms:.1f} ms", flush=True)

        print("\nTable 1A (Stride Sensitivity)")
        print("| stride | Offset (B) | Median (ms) | IQR (ms) | Mops/s |")
        print("|---|---|---|---|---|")
        for s, off, med, iqr, mops in t1a:
            print(f"| {s} | {off} | {med:.1f} | {iqr:.1f} | {mops:.2f} |")

        best_drop, drop_at = 0.0, None
        for prev, cur in zip(t1a, t1a[1:]):
            drop = prev[2] / cur[2]
            if drop > best_drop:
                best_drop, drop_at = drop, cur
        print(f"\nLargest drop: to stride={drop_at[0]} ({drop_at[1]} B offset), "
              f"time falls {best_drop:.2f}x versus the previous stride")

        med_s1 = t1a[0][2]
        med_s16 = t1a[4][2]

        # Table 1B: process scaling for stride 1 and 16
        ps = scaling_values(p_log)
        print(f"\nProcess scaling sweep, p in {ps}")
        rows1 = scaling_table(shm, ps, 1, m_total)
        rows16 = scaling_table(shm, ps, 16, m_total)
        print_scaling("Table 1B: stride = 1 (8 B, false sharing)", rows1)
        print_scaling("Table 1B: stride = 16 (128 B, isolated)", rows16)

        print()
        print(emit_verify_tag("PHASE1", args.student_id, med_s1, med_s16))
    finally:
        shm.close()
        shm.unlink()
