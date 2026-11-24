
import hashlib
import numpy as np
from numba import cuda, uint32

ROUND_CONSTANTS = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
], dtype=np.uint32)

@cuda.jit(device=True)
def rotr(x, n):
    return (x >> n) | (x << (32 - n))


@cuda.jit(device=True)
def sha256_device(message, output, length):

    k = cuda.const.array_like(ROUND_CONSTANTS)
    w = cuda.local.array(64, dtype=uint32)

    for i in range(16):
        w[i] = uint32(0)

    for i in range(length):
        j = i // 4
        shift = 24 - 8 * (i % 4)
        w[j] |= uint32(message[i]) << shift

    w[length // 4] |= uint32(0x80) << (24 - 8 * (length % 4))
    w[15] = uint32(length * 8)

    for i in range(16, 64):
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w[i] = uint32(w[i-16] + s0 + w[i-7] + s1)

    a = uint32(0x6a09e667)
    b = uint32(0xbb67ae85)
    c = uint32(0x3c6ef372)
    d = uint32(0xa54ff53a)
    e = uint32(0x510e527f)
    f = uint32(0x9b05688c)
    g = uint32(0x1f83d9ab)
    h = uint32(0x5be0cd19)

    for i in range(64):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        t1 = uint32(h + S1 + ch + k[i] + w[i])

        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = uint32(S0 + maj)

        h = g
        g = f
        f = e
        e = uint32(d + t1)
        d = c
        c = b
        b = a
        a = uint32(t1 + t2)

    output[0] = uint32(0x6a09e667 + a)
    output[1] = uint32(0xbb67ae85 + b)
    output[2] = uint32(0x3c6ef372 + c)
    output[3] = uint32(0xa54ff53a + d)
    output[4] = uint32(0x510e527f + e)
    output[5] = uint32(0x9b05688c + f)
    output[6] = uint32(0x1f83d9ab + g)
    output[7] = uint32(0x5be0cd19 + h)


@cuda.jit
def sha256_test_kernel(message, output):
    if cuda.grid(1) == 0:
        sha256_device(message, output, message.size)
        
if __name__ == "__main__":
   
    password = "maze"
    message = np.frombuffer(password.encode("ascii"), dtype=np.uint8)

    d_message = cuda.to_device(message)
    d_output = cuda.device_array(8, dtype=np.uint32)

    sha256_test_kernel[1, 1](d_message, d_output)

    gpu_hash = "".join(
        f"{int(x):08x}" for x in d_output.copy_to_host()
    )

    cpu_hash = hashlib.sha256(password.encode()).hexdigest()

    print("GPU SHA-256:", gpu_hash)
    print("CPU SHA-256:", cpu_hash)
    print("Hashes match:", gpu_hash == cpu_hash)

    assert gpu_hash == cpu_hash