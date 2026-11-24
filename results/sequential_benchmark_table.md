# CSI 4650 — Sequential CPU SHA-256 Benchmarks

All combinations were hashed and compared using Python's hashlib and itertools; no search stops early. CPU time is the measured search loop; target-hash setup and file writes are excluded.

| Workload | Combinations | Runs | Avg CPU (s) | SD (s) | M hashes/s | Correct |
|---|---:|---:|---:|---:|---:|---|
| lowercase_5 | 11,881,376 | 3 | 8.3729 | 0.2970 | 1.42 | Yes |
| lowercase_digits_5 | 60,466,176 | 3 | 42.6461 | 2.0230 | 1.42 | Yes |
