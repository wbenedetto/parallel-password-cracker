import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numba import cuda, uint8, uint32
from cuda_sha256 import sha256_device

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
BEST_SETTINGS = {
    ("lowercase", 5): (128, 1_000_000),
    ("lowercase", 6): (512, 10_000_000),
    ("lowercase_digits", 5): (128, 20_000_000),
    ("lowercase_digits", 6): (512, 20_000_000),
}
FALLBACK_SETTINGS = (512, 10_000_000)
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "optimized_cuda_runs.csv"

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


def password_index(password, charset):
    index = 0
    for character in password:
        index = index * len(charset) + charset.index(character)
    return index


def save_result(data):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = RESULTS_PATH.is_file()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data))
        if not existing:
            writer.writeheader()
        writer.writerow(data)

def main():
    if not cuda.is_available():
        raise RuntimeError("Numba-CUDA cannot access the NVIDIA GPU")

    target_password = input("Enter password to crack: ")

    PASSWORD_LENGTH = len(target_password)

    if not 4 <= PASSWORD_LENGTH <= 8:
        raise ValueError("Password must be between 4 and 8 characters")

    if not all(c in CHARSETS["lowercase_digits"] for c in target_password):
        raise ValueError(
            "Password can only contain lowercase letters (a-z) and digits (0-9)"
        )

    if all(c in CHARSETS["lowercase"] for c in target_password):
        CHARSET_NAME = "lowercase"
    else:
        CHARSET_NAME = "lowercase_digits"

    charset = CHARSETS[CHARSET_NAME]


    total = len(charset) ** PASSWORD_LENGTH
    expected_index = password_index(target_password, charset)
    (threads_per_block, batch_size) = BEST_SETTINGS.get(
        (CHARSET_NAME, PASSWORD_LENGTH), FALLBACK_SETTINGS
    )
    measured = (CHARSET_NAME, PASSWORD_LENGTH) in BEST_SETTINGS

    characters = np.frombuffer(charset.encode("ascii"), dtype=np.uint8)
    target_digest = hashlib.sha256(target_password.encode("ascii")).digest()
    target_words = np.array(
        [int.from_bytes(target_digest[i:i + 4], "big") for i in range(0, 32, 4)],
        dtype=np.uint32,
    )

    warmup_chars = cuda.to_device(characters)
    warmup_hash = cuda.to_device(target_words)
    warmup_result = cuda.to_device(np.array([total], dtype=np.int64))
    crack_kernel[1, 1](
        warmup_chars, warmup_hash, warmup_result,
        PASSWORD_LENGTH, np.int64(0), np.int64(1),
    )
    cuda.synchronize()

    overall_start = time.perf_counter()
    d_characters = cuda.to_device(characters)
    d_target = cuda.to_device(target_words)
    d_found = cuda.to_device(np.array([total], dtype=np.int64))
    cuda.synchronize()

    gpu_start = time.perf_counter()
    batch_count_total = 0
    for batch_start in range(0, total, batch_size):
        count = min(batch_size, total - batch_start)
        blocks = (count + threads_per_block - 1) // threads_per_block
        crack_kernel[blocks, threads_per_block](
            d_characters, d_target, d_found,
            PASSWORD_LENGTH, np.int64(batch_start), np.int64(count),
        )
        batch_count_total += 1

    cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start

    found_index = int(d_found.copy_to_host()[0])
    e2e_seconds = time.perf_counter() - overall_start
    correct = found_index == expected_index
    if not correct:
        raise RuntimeError(
            f"Incorrect result: expected index {expected_index}, got {found_index}"
        )

    print("\n=== OPTIMIZED CUDA SHA-256 ===")
    print(f"Workload:           {CHARSET_NAME}_{PASSWORD_LENGTH}")
    print(f"Password found:     {target_password}")
    print(f"Index:              {found_index:,}")
    print(f"Correct:            {correct}")
    print(f"Total combinations: {total:,}")
    print(f"Threads/block:      {threads_per_block}")
    print(f"Batch size:         {batch_size:,}")
    print(f"Batches:            {batch_count_total:,}")
    print(f"Settings source:    {'best measured' if measured else 'untested fallback'}")
    print(f"GPU kernel time:    {gpu_seconds:.6f}s")
    print(f"GPU end-to-end:     {e2e_seconds:.6f}s")
    print(f"GPU throughput:     {total / gpu_seconds:,.0f} hashes/sec")
    print(f"E2E throughput:     {total / e2e_seconds:,.0f} hashes/sec")

    save_result({
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Workload": f"{CHARSET_NAME}_{PASSWORD_LENGTH}",
        "Combinations": total,
        "Threads per block": threads_per_block,
        "Batch size": batch_size,
        "Batches": batch_count_total,
        "Settings measured": measured,
        "GPU seconds": gpu_seconds,
        "End-to-end seconds": e2e_seconds,
        "GPU million hashes/sec": total / gpu_seconds / 1e6,
        "Correct": correct,
    })
    print(f"Saved result:       {RESULTS_PATH}")


if __name__ == "__main__":
    main()
