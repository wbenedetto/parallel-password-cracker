import csv
import hashlib
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import cuda, uint8, uint32
from cuda_sha256 import sha256_device

THREAD_CONFIGS = [128, 256, 512, 1024]
BATCH_SIZES = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
PASSWORD_LENGTHS = [5, 6]
CHARSETS = {
    "lowercase": "abcdefghijklmnopqrstuvwxyz",
    "lowercase_digits": "abcdefghijklmnopqrstuvwxyz0123456789",
}
TARGETS = {
    5: "mazes",
    6: "mazels",
    7: "mazelsa",
}
REPEATS = 3
RANDOM_SEED = 4650

INCLUDE_LOWERCASE_7 = False

OUTPUT_FOLDER = Path("results")

@cuda.jit
def crack_kernel(characters, target_hash, found_index,
                 password_length, batch_start, batch_count):
    idx = cuda.grid(1)
    if idx >= batch_count:
        return

    global_index = batch_start + idx
    password = cuda.local.array(8, dtype=uint8)
    password_hash = cuda.local.array(8, dtype=uint32)

    value = global_index
    base = characters.size
    for position in range(password_length - 1, -1, -1):
        password[position] = characters[value % base]
        value //= base

    sha256_device(password, password_hash, password_length)

    match = True
    for i in range(8):
        if password_hash[i] != target_hash[i]:
            match = False
            break
    if match:
        cuda.atomic.min(found_index, 0, global_index)


def expected_password_index(password, charset):
    index = 0
    for char in password:
        index = index * len(charset) + charset.index(char)
    return index


def prepare_workload(charset_name, charset, length):
    password = TARGETS[length]
    if not (1 <= length <= 8):
        raise ValueError("The current CUDA password buffer supports lengths 1–8")
    if len(password) != length or any(c not in charset for c in password):
        raise ValueError(f"Target {password!r} is invalid for {charset_name} / {length}")
    if not charset.isascii() or len(set(charset)) != len(charset):
        raise ValueError("Character sets must be unique ASCII characters")

    target_digest = hashlib.sha256(password.encode("ascii")).digest()
    target_words = np.array([
        int.from_bytes(target_digest[i:i + 4], "big")
        for i in range(0, 32, 4)
    ], dtype=np.uint32)

    return {
        "name": f"{charset_name}_{length}",
        "charset_name": charset_name,
        "charset_size": len(charset),
        "password_length": length,
        "total": len(charset) ** length,
        "expected_index": expected_password_index(password, charset),
        "host_characters": np.frombuffer(charset.encode("ascii"), dtype=np.uint8),
        "host_target": target_words,
    }


def warmup(workload):
    """Compile the kernel and SHA-256 device function outside measurements."""
    d_characters = cuda.to_device(workload["host_characters"])
    d_target = cuda.to_device(workload["host_target"])
    d_found = cuda.to_device(np.array([workload["total"]], dtype=np.int64))
    crack_kernel[1, 1](
        d_characters, d_target, d_found,
        workload["password_length"], np.int64(0), np.int64(1)
    )
    cuda.synchronize()


def benchmark_once(workload, threads_per_block, batch_size):
    """Return launch+kernel time and end-to-end GPU-call time, in seconds.

    End-to-end includes GPU allocations, host-device copies, kernel launches,
    synchronization and copying the match index back. JIT compilation is excluded.
    """
    total = workload["total"]
    overall_start = time.perf_counter()

    d_characters = cuda.to_device(workload["host_characters"])
    d_target = cuda.to_device(workload["host_target"])
    d_found = cuda.to_device(np.array([total], dtype=np.int64))
    cuda.synchronize()

    gpu_start = time.perf_counter()
    batches = 0
    for batch_start in range(0, total, batch_size):
        batch_count = min(batch_size, total - batch_start)
        blocks_per_grid = (batch_count + threads_per_block - 1) // threads_per_block
        crack_kernel[blocks_per_grid, threads_per_block](
            d_characters,
            d_target,
            d_found,
            workload["password_length"],
            np.int64(batch_start),
            np.int64(batch_count),
        )
        batches += 1

    cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start

    found_index = int(d_found.copy_to_host()[0])
    overall_seconds = time.perf_counter() - overall_start

    if found_index != workload["expected_index"]:
        raise RuntimeError(
            f"Correctness failure: {workload['name']}, "
            f"threads={threads_per_block}, batch={batch_size:,}, "
            f"expected index={workload['expected_index']}, got={found_index}"
        )

    return gpu_seconds, overall_seconds, batches


def main():
    if not cuda.is_available():
        raise RuntimeError("Numba-CUDA cannot access an NVIDIA GPU")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_FOLDER / "mega_benchmark_raw.csv"
    summary_path = OUTPUT_FOLDER / "mega_benchmark_summary.csv"
    markdown_path = OUTPUT_FOLDER / "mega_benchmark_table.md"

    workloads = [
        prepare_workload(name, charset, length)
        for name, charset in CHARSETS.items()
        for length in PASSWORD_LENGTHS
    ]
    if INCLUDE_LOWERCASE_7:
        workloads.append(prepare_workload("lowercase", CHARSETS["lowercase"], 7))

    configs = [(threads, batch) for threads in THREAD_CONFIGS for batch in BATCH_SIZES]
    total_runs = len(workloads) * len(configs) * REPEATS
    print(f"CUDA benchmark: {len(workloads)} workloads × {len(configs)} configs "
          f"× {REPEATS} repeats = {total_runs} measured runs")
    print("Each run exhausts its workload, even after the target is found.\n")

    rng = random.Random(RANDOM_SEED)
    all_rows = []
    raw_fields = [
        "Workload", "Charset size", "Password length", "Combinations",
        "Run", "Threads per block", "Batch size", "Number of batches",
        "GPU seconds", "End-to-end seconds", "GPU million passwords/sec", "Correct",
    ]

    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        progress = 0

        for workload in workloads:
            print(f"--- {workload['name']}: {workload['total']:,} candidates ---")
            warmup(workload)

            for repetition in range(1, REPEATS + 1):
                shuffled = configs.copy()
                rng.shuffle(shuffled)
                for threads, batch in shuffled:
                    gpu_time, overall_time, number_of_batches = benchmark_once(
                        workload, threads, batch
                    )
                    row = {
                        "Workload": workload["name"],
                        "Charset size": workload["charset_size"],
                        "Password length": workload["password_length"],
                        "Combinations": workload["total"],
                        "Run": repetition,
                        "Threads per block": threads,
                        "Batch size": batch,
                        "Number of batches": number_of_batches,
                        "GPU seconds": gpu_time,
                        "End-to-end seconds": overall_time,
                        "GPU million passwords/sec": workload["total"] / gpu_time / 1e6,
                        "Correct": True,
                    }
                    writer.writerow(row)
                    handle.flush()
                    all_rows.append(row)
                    progress += 1
                    print(f"[{progress:3d}/{total_runs}] "
                          f"{workload['name']:<19}  "
                          f"T={threads:4d}  B={batch / 1e6:>4g}M  "
                          f"GPU={gpu_time:.4f}s  E2E={overall_time:.4f}s")

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["Workload"], row["Threads per block"], row["Batch size"])].append(row)

    summaries = []
    for (name, threads, batch), measurements in grouped.items():
        gpu_times = [m["GPU seconds"] for m in measurements]
        e2e_times = [m["End-to-end seconds"] for m in measurements]
        total = measurements[0]["Combinations"]
        mean_gpu = statistics.mean(gpu_times)
        summaries.append({
            "Workload": name,
            "Charset size": measurements[0]["Charset size"],
            "Password length": measurements[0]["Password length"],
            "Combinations": total,
            "Threads per block": threads,
            "Batch size": batch,
            "Runs": len(measurements),
            "Avg GPU seconds": mean_gpu,
            "SD GPU seconds": statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0.0,
            "Avg end-to-end seconds": statistics.mean(e2e_times),
            "GPU million passwords/sec": total / mean_gpu / 1e6,
        })

    summaries.sort(key=lambda s: (s["Workload"], s["Avg GPU seconds"]))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    markdown_lines = [
        "# CSI 4650 — CUDA Benchmark Grid",
        "",
        "GPU = warmed-up launch + kernel completion time; "
        "end-to-end = device allocations/transfers + launch + completion + result copy. "
        "Compilation is excluded from both.",
        "",
    ]

    print("\n=== AVERAGE RESULTS (ranked within each workload) ===")
    for workload in workloads:
        name = workload["name"]
        subset = [s for s in summaries if s["Workload"] == name]
        print(f"\n{name} — {workload['total']:,} candidates")
        print(" Rank  Threads   Batch   Avg GPU(s)  SD(s)   Avg E2E(s)    M pwd/s")
        print("-" * 69)
        markdown_lines.extend([
            f"## {name} — {workload['total']:,} candidates", "",
            "| Rank | Threads | Batch size | Avg GPU (s) | SD (s) | Avg E2E (s) | M passwords/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(subset, 1):
            threads = row["Threads per block"]
            batch = row["Batch size"]
            t = row["Avg GPU seconds"]
            sd = row["SD GPU seconds"]
            e2e = row["Avg end-to-end seconds"]
            rate = row["GPU million passwords/sec"]
            print(f" {rank:>3}  {threads:>7}  {batch / 1e6:>5g}M  "
                  f"{t:>10.4f}  {sd:>6.4f}  {e2e:>11.4f}  {rate:>9.1f}")
            markdown_lines.append(
                f"| {rank} | {threads} | {batch:,} | {t:.4f} | "
                f"{sd:.4f} | {e2e:.4f} | {rate:.1f} |"
            )
        markdown_lines.append("")
        winner = subset[0]
        print(f"Best in this sample: {winner['Threads per block']} threads/block, "
              f"{winner['Batch size']:,} candidates/batch")

    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print("\nSaved files:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {markdown_path}")

if __name__ == "__main__":
    main()
