import itertools
import hashlib
import time

CHARSET = "abcdefghijklmnopqrstuvwxyz"
PASSWORD_LENGTH = 5
TARGET_PASSWORD = "mazes"
BENCHMARK = True

target_hash = hashlib.sha256(TARGET_PASSWORD.encode()).digest()

def crack_password():
    attempts = 0
    found = None

    start = time.perf_counter()

    for combination in itertools.product(CHARSET, repeat=PASSWORD_LENGTH):
        password = "".join(combination)
        password_hash = hashlib.sha256(password.encode()).digest()

        attempts += 1

        if password_hash == target_hash:
            found = password

            if not BENCHMARK:
                break

    elapsed = time.perf_counter() - start

    print(f"Benchmark mode: {BENCHMARK}")
    print(f"Password: {found}")
    print(f"Attempts: {attempts:,}")
    print(f"Execution time: {elapsed:.4f} seconds")
    print(f"Passwords/sec: {attempts / elapsed:,.0f}")

crack_password()
