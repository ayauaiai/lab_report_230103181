import sys
import time
import multiprocessing as mp

from harness import emit_verify_tag

N_READS = 2_000_000  # 2 x 10^6 clock reads


def measure_clock_latency_ns(n: int = N_READS):
    """Average cost of one time.perf_counter_ns() call, in ns.

    Returns (avg_ns_per_call, total_elapsed_ms).
    """
    clock = time.perf_counter_ns
    for _ in range(10_000):  # warm-up
        clock()
    t0 = clock()
    for _ in range(n):
        clock()
    t1 = clock()
    total_ns = t1 - t0
    return total_ns / n, total_ns / 1e6


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python task0_cal.py <StudentID_integer>")
        sys.exit(1)
    student_seed = int(sys.argv[1])

    p_log = mp.cpu_count()
    avg_ns, total_ms = measure_clock_latency_ns()

    print(f"mp.cpu_count()           : {p_log}")
    print(f"Clock reads              : {N_READS}")
    print(f"Avg perf_counter_ns call : {avg_ns:.2f} ns")
    print(f"Total elapsed            : {total_ms:.2f} ms")
    print(emit_verify_tag("PHASE0", student_seed, avg_ns, total_ms))
