import argparse
import csv
import hashlib
import itertools
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

CHARSETS = {
    "lowercase": "abcdefghijklmnopqrstuvwxyz",
    "lowercase_digits": "abcdefghijklmnopqrstuvwxyz0123456789",
}
TARGETS = {
    4: "maze",
    5: "mazes",
    6: "mazels",
    7: "mazelsa",
    8: "mazelsab",
}

DEFAULT_LENGTHS = [5]
DEFAULT_REPEATS = 3
RANDOM_SEED = 4650
OUTPUT_FOLDER = Path("results")


def expected_password_index(password, charset):
    """Zero-based location of a password in itertools.product order."""
    index = 0
    for char in password:
        index = index * len(charset) + charset.index(char)
    return index


def make_workload(charset_name, charset, length):
    target_password = TARGETS[length]
    if len(set(charset)) != len(charset) or not charset.isascii():
        raise ValueError("Character set must contain unique ASCII characters")
    if len(target_password) != length or any(c not in charset for c in target_password):
        raise ValueError(f"Target {target_password!r} is invalid for {charset_name}_{length}")

    return {
        "name": f"{charset_name}_{length}",
        "charset_name": charset_name,
        "charset": charset,
        "charset_size": len(charset),
        "password_length": length,
        "target": target_password,
        "target_hash": hashlib.sha256(target_password.encode("ascii")).digest(),
        "expected_index": expected_password_index(target_password, charset),
        "total": len(charset) ** length,
    }


def benchmark_once(workload):

    found_index = -1
    attempts = 0
    start = time.perf_counter()

    for index, chars in enumerate(
        itertools.product(workload["charset"], repeat=workload["password_length"])
    ):
        password = "".join(chars)
        candidate_hash = hashlib.sha256(password.encode("ascii")).digest()
        attempts += 1
        if candidate_hash == workload["target_hash"]:
            found_index = index

    elapsed = time.perf_counter() - start
    if attempts != workload["total"] or found_index != workload["expected_index"]:
        raise RuntimeError(
            f"Correctness failure in {workload['name']}: "
            f"processed={attempts:,}, found_index={found_index}, "
            f"expected={workload['expected_index']}"
        )
    return elapsed, found_index, attempts


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential CPU SHA-256 benchmark")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                        help="Password lengths (default: 5). Use 4 for a quick test.")
    parser.add_argument("--charsets", nargs="+", choices=list(CHARSETS),
                        default=list(CHARSETS), help="Character sets to benchmark")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Number of runs per workload (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_FOLDER,
                        help="Folder for CSV and Markdown results")
    args = parser.parse_args()
    if args.repeats < 1 or any(n not in TARGETS for n in args.lengths):
        parser.error(f"--repeats must be >= 1, and lengths must be one of {list(TARGETS)}")
    return args


def main():
    args = parse_args()
    workloads = [
        make_workload(name, CHARSETS[name], length)
        for name in args.charsets
        for length in args.lengths
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sequential_benchmark_raw.csv"
    summary_path = args.output_dir / "sequential_benchmark_summary.csv"
    table_path = args.output_dir / "sequential_benchmark_table.md"

    print(f"CPU sequential benchmark: {len(workloads)} workloads x "
          f"{args.repeats} repetitions = {len(workloads) * args.repeats} runs")
    print("All runs exhaust their full search spaces; no early exit on match.\n")
    for w in workloads:
        print(f"  {w['name']:<22} {w['total']:>15,} candidates/run")
    print()
    if any(w["total"] > 100_000_000 for w in workloads):
        print("NOTE: Some selected CPU workloads exceed 100 million hashes per run.\n")

    raw_fields = [
        "Workload", "Charset size", "Password length", "Combinations",
        "Run", "CPU seconds", "CPU million passwords/sec", "Found index", "Correct",
    ]
    rows = []
    rng = random.Random(RANDOM_SEED)

    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        for run in range(1, args.repeats + 1):
            order = workloads.copy()
            rng.shuffle(order)
            for workload in order:
                elapsed, found_index, attempts = benchmark_once(workload)
                row = {
                    "Workload": workload["name"],
                    "Charset size": workload["charset_size"],
                    "Password length": workload["password_length"],
                    "Combinations": attempts,
                    "Run": run,
                    "CPU seconds": elapsed,
                    "CPU million passwords/sec": attempts / elapsed / 1e6,
                    "Found index": found_index,
                    "Correct": True,
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(f"Run {run} | {workload['name']:<22} | "
                      f"{elapsed:>9.4f}s | {attempts / elapsed / 1e6:>7.2f}M hashes/sec | PASS")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Workload"]].append(row)

    summary = []
    for workload in workloads:
        measurements = grouped[workload["name"]]
        times = [r["CPU seconds"] for r in measurements]
        avg = statistics.mean(times)
        summary.append({
            "Workload": workload["name"],
            "Charset size": workload["charset_size"],
            "Password length": workload["password_length"],
            "Combinations": workload["total"],
            "Runs": len(times),
            "Avg CPU seconds": avg,
            "SD CPU seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "CPU million passwords/sec": workload["total"] / avg / 1e6,
            "Correct": True,
        })

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    table_lines = [
        "# CSI 4650 — Sequential CPU SHA-256 Benchmarks", "",
        "All combinations were hashed and compared using Python's hashlib and itertools; "
        "no search stops early. CPU time is the measured search loop; "
        "target-hash setup and file writes are excluded.", "",
        "| Workload | Combinations | Runs | Avg CPU (s) | SD (s) | M hashes/s | Correct |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    print("\n=== SEQUENTIAL CPU SUMMARY ===")
    print(f"{'Workload':<22} {'Combinations':>15} {'Avg CPU(s)':>12} {'SD(s)':>10} {'M/sec':>9}")
    print("-" * 74)
    for result in summary:
        print(f"{result['Workload']:<22} {result['Combinations']:>15,} "
              f"{result['Avg CPU seconds']:>12.4f} "
              f"{result['SD CPU seconds']:>10.4f} "
              f"{result['CPU million passwords/sec']:>9.2f}")
        table_lines.append(
            f"| {result['Workload']} | {result['Combinations']:,} | {result['Runs']} | "
            f"{result['Avg CPU seconds']:.4f} | {result['SD CPU seconds']:.4f} | "
            f"{result['CPU million passwords/sec']:.2f} | Yes |"
        )
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"\nSaved:\n  {raw_path}\n  {summary_path}\n  {table_path}")


if __name__ == "__main__":
    main()
