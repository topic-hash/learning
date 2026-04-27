#!/usr/bin/env python3
"""
SHA-256 Round 16 Shortcut using standard Boolean/Bitwise Operations.
Numba-accelerated for performance comparison with Tropical variant.
"""

import numpy as np
from numba import njit
from struct import pack

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
def rotr32(x, n):
    """Rotate right a 32-bit integer by n positions"""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


@njit
def shr32(x, n):
    """Shift right a 32-bit integer by n positions"""
    return x >> n


@njit
def ch(x, y, z):
    """Ch(x, y, z) = (x AND y) XOR (NOT x AND z)"""
    return (x & y) ^ (~x & z)


@njit
def maj(x, y, z):
    """Maj(x, y, z) = majority vote"""
    return (x & y) ^ (x & z) ^ (y & z)


@njit
def sigma0(x):
    """Σ0(x) = rotr(x,2) XOR rotr(x,13) XOR rotr(x,22)"""
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)


@njit
def sigma1(x):
    """Σ1(x) = rotr(x,6) XOR rotr(x,11) XOR rotr(x,25)"""
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)


@njit
def gamma0(x):
    """σ0(x) = rotr(x,7) XOR rotr(x,18) XOR shr(x,3)"""
    return rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)


@njit
def gamma1(x):
    """σ1(x) = rotr(x,17) XOR rotr(x,19) XOR shr(x,10)"""
    return rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)


@njit
def sha256_bool_rounds(W, H, K, num_rounds):
    """
    Perform standard boolean SHA-256 rounds up to num_rounds.
    W: message schedule (64 uint32 values)
    H: hash state (8 uint32 values, modified in place)
    K: K constants array
    num_rounds: number of rounds to execute
    """
    # Working variables (copies of H)
    a = H[0]
    b = H[1]
    c = H[2]
    d = H[3]
    e = H[4]
    f = H[5]
    g = H[6]
    h = H[7]
    
    for i in range(num_rounds):
        if i >= 16:
            # Expand message schedule
            W[i] = np.uint32(gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16])
        
        # T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
        T1 = np.uint32(h + sigma1(e) + ch(e, f, g) + K[i] + W[i])
        
        # T2 = Σ0(a) + Maj(a,b,c)
        T2 = np.uint32(sigma0(a) + maj(a, b, c))
        
        # Update working variables
        h = g
        g = f
        f = e
        e = np.uint32(d + T1)
        d = c
        c = b
        b = a
        a = np.uint32(T1 + T2)
    
    # Add working variables back to hash state
    H[0] = np.uint32(H[0] + a)
    H[1] = np.uint32(H[1] + b)
    H[2] = np.uint32(H[2] + c)
    H[3] = np.uint32(H[3] + d)
    H[4] = np.uint32(H[4] + e)
    H[5] = np.uint32(H[5] + f)
    H[6] = np.uint32(H[6] + g)
    H[7] = np.uint32(H[7] + h)


def compute_sha256_bool_round16(message_bytes):
    """
    Compute standard boolean SHA-256 intermediate state after round 16.
    
    Args:
        message_bytes: Input message as bytes
    
    Returns:
        List of 8 uint32 values representing intermediate hash state
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
    
    # Initialize message schedule W (64 uint32 values)
    W = np.empty(64, dtype=np.uint32)
    
    # Parse first 16 words from padded message
    for i in range(16):
        word = (int(padded[i*4]) << 24) | (int(padded[i*4+1]) << 16) | \
               (int(padded[i*4+2]) << 8) | int(padded[i*4+3])
        W[i] = word & 0xFFFFFFFF
    
    # Initialize hash state H (8 uint32 values)
    H = H_INIT.copy()
    
    # Execute 16 rounds
    sha256_bool_rounds(W, H, K_INIT, 16)
    
    return [H[i] for i in range(8)]


def format_hash(state):
    """Format hash state as hex string."""
    return ''.join(f'{x:08x}' for x in state)


if __name__ == "__main__":
    print("Standard Boolean SHA-256 - Intermediate State After Round 16")
    print("=" * 70)
    
    test_messages = [
        b"",
        b"abc",
        b"The quick brown fox jumps over the lazy dog",
        b"a" * 55,
    ]
    
    for msg in test_messages:
        try:
            state = compute_sha256_bool_round16(msg)
            print(f"\nMessage: {msg[:40]}{'...' if len(msg) > 40 else ''}")
            print(f"Length: {len(msg)} bytes")
            print(f"Boolean intermediate state: {format_hash(state)}")
            print(f"State words: {[f'0x{x:08x}' for x in state]}")
        except Exception as e:
            print(f"\nMessage: {msg[:40]}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("This is standard SHA-256 with bitwise boolean operations.")
