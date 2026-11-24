
import hashlib
import time
import numpy as np
from numba import cuda, uint8, uint32
from cuda_sha256 import sha256_device

CHARSET = "abcdefghijklmnopqrstuvwxyz"
PASSWORD_LENGTH = 7
TARGET_PASSWORD = "mazelsa"

THREADS_PER_BLOCK = 1024
BATCH_SIZE = 10_000_000

TOTAL_COMBINATIONS = len(CHARSET) ** PASSWORD_LENGTH

assert 1 <= PASSWORD_LENGTH <= 8
assert len(TARGET_PASSWORD) == PASSWORD_LENGTH
assert all(c in CHARSET for c in TARGET_PASSWORD)


@cuda.jit
def crack_kernel(characters, target_hash, found_index,
                 password_length, batch_start, batch_count):

    idx = cuda.grid(1)

    if idx >= batch_count:
        return

    global_index = batch_start + idx

    password = cuda.local.array(8, dtype=uint8)
    password_hash = cuda.local.array(8, dtype=uint32)

    number = global_index
    base = characters.size

    for position in range(password_length - 1, -1, -1):
        char_index = number % base
        password[position] = characters[char_index]
        number //= base

    sha256_device(password, password_hash, password_length)

    match = True

    for i in range(8):
        if password_hash[i] != target_hash[i]:
            match = False
            break

    if match:
        cuda.atomic.min(found_index, 0, global_index)

characters = np.frombuffer(
    CHARSET.encode("ascii"), dtype=np.uint8
)

target = hashlib.sha256(
    TARGET_PASSWORD.encode("ascii")
).digest()

target_words = np.array([
    int.from_bytes(target[i:i+4], "big")
    for i in range(0, 32, 4)
], dtype=np.uint32)

d_characters = cuda.to_device(characters)
d_target = cuda.to_device(target_words)

d_warmup = cuda.to_device(
    np.array([TOTAL_COMBINATIONS], dtype=np.int64)
)

crack_kernel[1, 1](
    d_characters, d_target, d_warmup,
    PASSWORD_LENGTH, np.int64(0), np.int64(1)
)
cuda.synchronize()

d_found = cuda.to_device(
    np.array([TOTAL_COMBINATIONS], dtype=np.int64)
)
start_time = time.perf_counter()

batches = 0

for batch_start in range(0, TOTAL_COMBINATIONS, BATCH_SIZE):

    batch_count = min(
        BATCH_SIZE,
        TOTAL_COMBINATIONS - batch_start
    )

    blocks = (
        batch_count + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    crack_kernel[blocks, THREADS_PER_BLOCK](
        d_characters,
        d_target,
        d_found,
        PASSWORD_LENGTH,
        np.int64(batch_start),
        np.int64(batch_count)
    )

    batches += 1

cuda.synchronize()

elapsed = time.perf_counter() - start_time

found_index = int(d_found.copy_to_host()[0])

if found_index < TOTAL_COMBINATIONS:
    number = found_index
    result = [""] * PASSWORD_LENGTH

    for position in range(PASSWORD_LENGTH - 1, -1, -1):
        result[position] = CHARSET[number % len(CHARSET)]
        number //= len(CHARSET)

    found_password = "".join(result)

    print(f"Password found: {found_password}")
    print(f"Index: {found_index:,}")
    print(f"Correct: {found_password == TARGET_PASSWORD}")

else:
    print("Password not found")

print(f"Total combinations: {TOTAL_COMBINATIONS:,}")
print(f"Batch size: {BATCH_SIZE:,}")
print(f"Batches processed: {batches:,}")
print(f"Threads per block: {THREADS_PER_BLOCK}")
print(f"Execution time: {elapsed:.4f} seconds")
print(f"Passwords/sec: {TOTAL_COMBINATIONS / elapsed:,.0f}")