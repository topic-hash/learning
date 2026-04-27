#!/usr/bin/env python3
"""
SHA-256 Round 16 Shortcut using Tropical Algebra (Min-Plus Semiring) in Floating Point.

This script implements a tropicalized variant of SHA-256 where all boolean/bitwise 
operations are replaced with Tropical Algebra operations over floating point numbers.

Tropical (Min-Plus) Semiring:
  - Tropical Addition (⊕): min(a, b)  -- analogous to OR
  - Tropical Multiplication (⊗): a + b  -- analogous to AND

Boolean Logic Mapping (True=0.0, False=C_TROP):
  - NOT(a)     = C_TROP - a
  - AND(a,b)   = a + b
  - OR(a,b)    = min(a, b)
  - XOR(a,b)   = min(a + (C-b), (C-a) + b)
  - Ch(x,y,z)  = min(x + y, (C-x) + z)  -- if x then y else z
  - Maj(x,y,z) = min(x+y, x+z, y+z)     -- at least two must be true

Each 32-bit word is represented as 32 floats (one per bit position).
"""

import numpy as np
from numba import njit
from struct import pack

# Tropical Constants
# True = 0.0 (Low Energy/Cost)
# False = 10.0 (High Energy/Cost)
C_TROP = 10.0

# SHA-256 Constants
K_INIT = np.array([
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

H_INIT = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
], dtype=np.uint32)


@njit
def trop_not(a):
    """Tropical NOT: C - a"""
    return C_TROP - a


@njit
def trop_and(a, b):
    """Tropical AND: a + b"""
    return a + b


@njit
def trop_or(a, b):
    """Tropical OR: min(a, b)"""
    return a if a < b else b


@njit
def trop_xor(a, b):
    """
    Tropical XOR: min(a + (C-b), (C-a) + b)
    Truth table verification:
      a=0, b=0 -> min(C, C) = C (False) ✓
      a=0, b=C -> min(0, 2C) = 0 (True) ✓
      a=C, b=0 -> min(2C, 0) = 0 (True) ✓
      a=C, b=C -> min(C, C) = C (False) ✓
    """
    term1 = a + (C_TROP - b)
    term2 = (C_TROP - a) + b
    return term1 if term1 < term2 else term2


@njit
def trop_ch(x, y, z):
    """
    Tropical Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
    Simplified: min(x + y, (C-x) + z)
    If x=0 (True): min(y, C+z) = y ✓
    If x=C (False): min(C+y, z) = z ✓
    """
    term1 = x + y
    term2 = (C_TROP - x) + z
    return term1 if term1 < term2 else term2


@njit
def trop_maj(x, y, z):
    """
    Tropical Maj(x, y, z) = majority vote
    min(x+y, x+z, y+z)
    At least two must be 0 (True) for result to be 0 (True)
    """
    v1 = x + y
    v2 = x + z
    v3 = y + z
    
    m = v1
    if v2 < m:
        m = v2
    if v3 < m:
        m = v3
    return m


@njit
def vec_trop_xor(a, b):
    """Vectorized tropical XOR for 32-float vectors"""
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        res[i] = trop_xor(a[i], b[i])
    return res


@njit
def vec_trop_ch(x, y, z):
    """Vectorized tropical Ch for 32-float vectors"""
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        res[i] = trop_ch(x[i], y[i], z[i])
    return res


@njit
def vec_trop_maj(x, y, z):
    """Vectorized tropical Maj for 32-float vectors"""
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        res[i] = trop_maj(x[i], y[i], z[i])
    return res


@njit
def vec_rotr(arr, n):
    """Rotate right a 32-float vector by n positions"""
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        res[(i + n) % 32] = arr[i]
    return res


@njit
def vec_shr(arr, n):
    """Shift right a 32-float vector by n positions, filling with C_TROP (False)"""
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        if i >= n:
            res[i] = arr[i - n]
        else:
            res[i] = C_TROP
    return res


@njit
def vec_sigma0(x):
    """Tropical Σ0: rotr(x,2) XOR rotr(x,13) XOR rotr(x,22)"""
    r2 = vec_rotr(x, 2)
    r13 = vec_rotr(x, 13)
    r22 = vec_rotr(x, 22)
    
    tmp = vec_trop_xor(r2, r13)
    return vec_trop_xor(tmp, r22)


@njit
def vec_sigma1(x):
    """Tropical Σ1: rotr(x,6) XOR rotr(x,11) XOR rotr(x,25)"""
    r6 = vec_rotr(x, 6)
    r11 = vec_rotr(x, 11)
    r25 = vec_rotr(x, 25)
    
    tmp = vec_trop_xor(r6, r11)
    return vec_trop_xor(tmp, r25)


@njit
def vec_gamma0(x):
    """Tropical σ0: rotr(x,7) XOR rotr(x,18) XOR shr(x,3)"""
    r7 = vec_rotr(x, 7)
    r18 = vec_rotr(x, 18)
    s3 = vec_shr(x, 3)
    
    tmp = vec_trop_xor(r7, r18)
    return vec_trop_xor(tmp, s3)


@njit
def vec_gamma1(x):
    """Tropical σ1: rotr(x,17) XOR rotr(x,19) XOR shr(x,10)"""
    r17 = vec_rotr(x, 17)
    r19 = vec_rotr(x, 19)
    s10 = vec_shr(x, 10)
    
    tmp = vec_trop_xor(r17, r19)
    return vec_trop_xor(tmp, s10)


@njit
def vec_add(a, b):
    """
    Element-wise tropical addition.
    In the tropical Min-Plus semiring used for mixing (like standard SHA256 +),
    we use standard float addition with clamping to stay within [0, C_TROP].
    """
    res = np.empty(32, dtype=np.float64)
    for i in range(32):
        val = a[i] + b[i]
        # Clamp to valid range - values can exceed C_TROP representing "more false"
        if val > 2 * C_TROP:
            val = 2 * C_TROP
        res[i] = val
    return res


def uint32_to_trop_vec(val):
    """Convert uint32 to 32-float tropical vector (1->0.0, 0->C_TROP)"""
    vec = np.empty(32, dtype=np.float64)
    for i in range(32):
        bit = (val >> (31 - i)) & 1
        vec[i] = 0.0 if bit == 1 else C_TROP
    return vec


def trop_vec_to_uint32(vec):
    """Convert 32-float tropical vector back to uint32 (threshold at C_TROP/2)"""
    val = 0
    for i in range(32):
        if vec[i] < (C_TROP / 2):
            val |= (1 << (31 - i))
    return val


@njit
def sha256_trop_rounds(W, H, K_trop, num_rounds):
    """
    Perform tropical SHA-256 rounds up to num_rounds.
    W: message schedule (64 x 32-float vectors)
    H: hash state (8 x 32-float vectors, modified in place)
    K_trop: pre-converted tropical K constants (64 x 32 floats)
    num_rounds: number of rounds to execute
    """
    # Working variables (copies of H)
    a = H[0].copy()
    b = H[1].copy()
    c = H[2].copy()
    d = H[3].copy()
    e = H[4].copy()
    f = H[5].copy()
    g = H[6].copy()
    h = H[7].copy()
    
    for i in range(num_rounds):
        if i >= 16:
            # Expand message schedule using tropical gamma functions
            # W[i] = γ1(W[i-2]) + W[i-7] + γ0(W[i-15]) + W[i-16]
            t1 = vec_gamma1(W[i-2])
            t2 = W[i-7]
            t3 = vec_gamma0(W[i-15])
            t4 = W[i-16]
            
            W[i] = vec_add(vec_add(t1, t2), vec_add(t3, t4))
        
        # T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
        S1 = vec_sigma1(e)
        Ch = vec_trop_ch(e, f, g)
        
        # Use pre-converted K constant
        K_const = K_trop[i]
        
        T1 = vec_add(vec_add(h, S1), vec_add(Ch, vec_add(K_const, W[i])))
        
        # T2 = Σ0(a) + Maj(a,b,c)
        S0 = vec_sigma0(a)
        Maj = vec_trop_maj(a, b, c)
        T2 = vec_add(S0, Maj)
        
        # Update working variables
        h = g.copy()
        g = f.copy()
        f = e.copy()
        e = vec_add(d, T1)
        d = c.copy()
        c = b.copy()
        b = a.copy()
        a = vec_add(T1, T2)
    
    # Add working variables back to hash state
    for i in range(8):
        H[i] = vec_add(H[i], [a, b, c, d, e, f, g, h][i])


def compute_sha256_trop_round16(message_bytes):
    """
    Compute tropical SHA-256 intermediate state after round 16.
    
    Args:
        message_bytes: Input message as bytes
    
    Returns:
        List of 8 uint32 values (converted back from tropical state)
    """
    if len(message_bytes) > 55:
        raise ValueError("Message too long for single-block processing")
    
    pad_length = 64
    
    # Prepare padded message
    padded = np.zeros(pad_length, dtype=np.uint8)
    for i in range(len(message_bytes)):
        padded[i] = message_bytes[i]
    
    if len(message_bytes) < pad_length:
        padded[len(message_bytes)] = 0x80
    
    msg_len_bits = len(message_bytes) * 8
    for i in range(8):
        padded[pad_length - 1 - i] = (msg_len_bits >> (i * 8)) & 0xFF
    
    # Initialize message schedule W (64 x 32 floats)
    W = np.empty((64, 32), dtype=np.float64)
    
    # Parse first 16 words from padded message into tropical vectors
    for i in range(16):
        word = (int(padded[i*4]) << 24) | (int(padded[i*4+1]) << 16) | \
               (int(padded[i*4+2]) << 8) | int(padded[i*4+3])
        W[i] = uint32_to_trop_vec(word & 0xFFFFFFFF)
    
    # Initialize hash state H (8 x 32 floats)
    H = np.empty((8, 32), dtype=np.float64)
    for i in range(8):
        H[i] = uint32_to_trop_vec(H_INIT[i])
    
    # Pre-convert K constants to tropical vectors
    K_trop = np.empty((64, 32), dtype=np.float64)
    for i in range(64):
        K_trop[i] = uint32_to_trop_vec(K_INIT[i])
    
    # Execute 16 rounds
    sha256_trop_rounds(W, H, K_trop, 16)
    
    # Convert back to uint32
    result = []
    for i in range(8):
        result.append(trop_vec_to_uint32(H[i]))
    
    return result


def format_hash(state):
    """Format hash state as hex string."""
    return ''.join(f'{x:08x}' for x in state)


if __name__ == "__main__":
    print("Tropical SHA-256 (Min-Plus Semiring) - Intermediate State After Round 16")
    print("=" * 70)
    print(f"Tropical Constant C = {C_TROP}")
    print(f"Mapping: True=0.0, False={C_TROP}")
    print(f"Operations: AND→+, OR→min, NOT→C-a, XOR→min(a+C-b,C-a+b)")
    print("=" * 70)
    
    test_messages = [
        b"",
        b"abc",
        b"The quick brown fox jumps over the lazy dog",
        b"a" * 55,
    ]
    
    for msg in test_messages:
        try:
            state = compute_sha256_trop_round16(msg)
            print(f"\nMessage: {msg[:40]}{'...' if len(msg) > 40 else ''}")
            print(f"Length: {len(msg)} bytes")
            print(f"Tropical intermediate state: {format_hash(state)}")
            print(f"State words: {[f'0x{x:08x}' for x in state]}")
        except Exception as e:
            print(f"\nMessage: {msg[:40]}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("Note: This is a TROPICALIZED SHA-256 using Min-Plus algebra.")
    print("All boolean operations replaced with floating point tropical ops.")
    print("Output is NOT a valid SHA-256 hash - for cryptanalysis/SAT only.")
