
import hashlib
import numpy as np
from numba import cuda, uint8, uint32
from cuda_sha256 import sha256_device
import time

CHARSET = "abcdefghijklmnopqrstuvwxyz"
PASSWORD_LENGTH = 7
TARGET_PASSWORD = "amazels"

TOTAL_COMBINATIONS = len(CHARSET) ** PASSWORD_LENGTH
THREADS_PER_BLOCK = 256


@cuda.jit
def crack_kernel(characters, target_hash, found_index,
                 password_length, total):

    idx = cuda.grid(1)

    if idx >= total:
        return

    password = cuda.local.array(8, dtype=uint8)
    password_hash = cuda.local.array(8, dtype=uint32)

    number = idx
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
        cuda.atomic.min(found_index, 0, idx)

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

d_found = cuda.to_device(
    np.array([TOTAL_COMBINATIONS], dtype=np.int64)
)

BLOCKS_PER_GRID = (
    TOTAL_COMBINATIONS + THREADS_PER_BLOCK - 1
) // THREADS_PER_BLOCK

crack_kernel[1, 1](
    d_characters, d_target, d_found,
    PASSWORD_LENGTH, TOTAL_COMBINATIONS
)
cuda.synchronize()

start = time.perf_counter()

crack_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
    d_characters, d_target, d_found,
    PASSWORD_LENGTH, TOTAL_COMBINATIONS
)

cuda.synchronize()
elapsed = time.perf_counter() - start

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
print(f"Threads per block: {THREADS_PER_BLOCK}")
print(f"Blocks per grid: {BLOCKS_PER_GRID}")
print(f"Execution time: {elapsed:.4f} seconds")
print(f"Passwords/sec: {TOTAL_COMBINATIONS / elapsed:,.0f}")