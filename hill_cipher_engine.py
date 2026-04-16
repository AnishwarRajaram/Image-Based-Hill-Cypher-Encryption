"""
Core Hill Cipher Engine  

Implements the Two-Stage Hill Cipher (TSHC) 

Public API

- generate_key(n, q)          → random n×n key from SL_n(F_q)
- block_divide(channel, n)    → (blocks, orig_shape, padded_shape)
- block_merge(blocks, orig_shape, padded_shape, n)
- encrypt_channel(channel, K, n, q, pre=True)
- decrypt_channel(cipher_channel, K, n, q, pre=True)

All arithmetic is mod q (prime field F_q).

Dependencies: numpy, sympy (for modular matrix inverse / SL_n sampling).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# Finite-field helpers

def _mod_det(M: np.ndarray, q: int) -> int:
    """Determinant of an integer matrix mod q (Bareiss-style, exact)."""
    A = M.astype(object) % q          # use Python ints to avoid overflow
    n = A.shape[0]
    sign = 1
    for col in range(n):
        # find pivot
        pivot_row = None
        for row in range(col, n):
            if int(A[row, col]) % q != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return 0
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            sign *= -1
        inv_pivot = pow(int(A[col, col]) % q, -1, q)   # modular inverse
        for row in range(col + 1, n):
            factor = int(A[row, col]) * inv_pivot % q
            A[row] = (A[row] - factor * A[col]) % q
    det = sign
    for i in range(n):
        det = det * int(A[i, i]) % q
    return int(det) % q


def _mod_inv_matrix(M: np.ndarray, q: int) -> np.ndarray:
    """
    Modular inverse of matrix M over F_q using Gauss-Jordan elimination.
    For keys in SL_n(F_q), inv(K) == adj(K) since det=1.
    Raises ValueError if M is singular mod q.
    """
    n = M.shape[0]
    A = np.hstack([M.astype(object) % q,
                   np.eye(n, dtype=object)])
    for col in range(n):
        pivot_row = None
        for row in range(col, n):
            if int(A[row, col]) % q != 0:
                pivot_row = row
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular mod q; cannot invert.")
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
        inv_pivot = pow(int(A[col, col]) % q, -1, q)
        A[col] = A[col] * inv_pivot % q
        for row in range(n):
            if row != col:
                factor = int(A[row, col])
                A[row] = (A[row] - factor * A[col]) % q
    return np.array(A[:, n:], dtype=np.int64)


def _mat_mul_mod(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    """Matrix multiplication mod q."""
    return (A.astype(object) @ B.astype(object) % q).astype(np.int64)


# Key generation from SL_n(F_q)

def generate_key(n: int, q: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a random n×n matrix in SL_n(F_q):
      det(K) ≡ 1  (mod q),  all entries in {0, …, q-1}.

    Strategy: draw a random invertible matrix, then scale the first row so
    that the determinant becomes 1.

    Parameters
    ----------
    n : block / key size
    q : prime modulus (field size)
    rng : optional numpy Generator for reproducibility

    Returns
    -------
    K : (n, n) int64 array, det(K) ≡ 1 mod q
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(10_000):          # retry until invertible
        K = rng.integers(0, q, size=(n, n), dtype=np.int64)
        d = _mod_det(K, q)
        if d == 0:
            continue
        # Scale row 0 by d^{-1} to make det = 1
        d_inv = pow(int(d), -1, q)
        K[0] = (K[0].astype(object) * d_inv % q).astype(np.int64)
        assert _mod_det(K, q) == 1, "det after scaling should be 1"
        return K

    raise RuntimeError(
        f"Could not generate an invertible {n}×{n} matrix over F_{q} "
        "after 10 000 attempts. Try a larger q."
    )


# Block division / merge  (with automatic zero-padding)


def block_divide(
    channel: np.ndarray,
    n: int,
) -> Tuple[List[np.ndarray], Tuple[int, int], Tuple[int, int]]:
    """
    Divide a 2D channel into non-overlapping n×n sub-blocks.

    If the channel dimensions are not divisible by n, zero-pad the right /
    bottom edges (the paper states redundant rows/columns are appended and
    stripped during decryption).

    Parameters
    ----------
    channel : (H, W) uint8 or int array
    n       : block size

    Returns
    -------
    blocks       : list of (n, n) int64 arrays, row-major order
    orig_shape   : (H, W) before padding
    padded_shape : (H', W') after padding (multiples of n)
    """
    orig_shape = channel.shape[:2]
    H, W = orig_shape
    H_pad = math.ceil(H / n) * n
    W_pad = math.ceil(W / n) * n
    padded = np.zeros((H_pad, W_pad), dtype=np.int64)
    padded[:H, :W] = channel.astype(np.int64)

    blocks: List[np.ndarray] = []
    for r in range(0, H_pad, n):
        for c in range(0, W_pad, n):
            blocks.append(padded[r:r+n, c:c+n].copy())

    return blocks, orig_shape, (H_pad, W_pad)


def block_merge(
    blocks: List[np.ndarray],
    orig_shape: Tuple[int, int],
    padded_shape: Tuple[int, int],
    n: int,
) -> np.ndarray:
    """
    Reassemble blocks into a 2D channel and strip the padding.

    Parameters
 
    blocks       : list of (n, n) arrays (same order as block_divide output)
    orig_shape   : (H, W) original dimensions
    padded_shape : (H', W') padded dimensions
    n            : block size

    Returns
    channel : (H, W) uint8 array
    """
    H_pad, W_pad = padded_shape
    canvas = np.zeros((H_pad, W_pad), dtype=np.int64)
    idx = 0
    for r in range(0, H_pad, n):
        for c in range(0, W_pad, n):
            canvas[r:r+n, c:c+n] = blocks[idx]
            idx += 1

    H, W = orig_shape
    return canvas[:H, :W]


# Single-block encrypt / decrypt

def _encrypt_block(B: np.ndarray, K: np.ndarray, q: int, pre: bool) -> np.ndarray:
    """
    C = K @ B  mod q   (pre-multiplication)  or
    C = B @ K  mod q   (post-multiplication)

    The paper emphasises that the position (pre / post) is part of the secret.
    """
    if pre:
        return _mat_mul_mod(K, B, q)
    else:
        return _mat_mul_mod(B, K, q)


def _decrypt_block(C: np.ndarray, K_inv: np.ndarray, q: int, pre: bool) -> np.ndarray:
    """
    B = K_inv @ C  mod q   (if encryption used pre-multiplication)
    B = C @ K_inv  mod q   (if encryption used post-multiplication)
    """
    if pre:
        return _mat_mul_mod(K_inv, C, q)
    else:
        return _mat_mul_mod(C, K_inv, q)


# Full-channel encrypt / decrypt

def encrypt_channel(
    channel: np.ndarray,
    K: np.ndarray,
    n: int,
    q: int,
    pre: bool = True,
) -> np.ndarray:
    """
    Apply Hill Cipher to one (H, W) channel.

    Parameters

    channel : (H, W) uint8 / int array — one colour plane
    K       : (n, n) key matrix from SL_n(F_q)
    n       : block size (must equal K.shape[0])
    q       : prime modulus
    pre     : True  → C = K @ B mod q  (pre-multiplication)
              False → C = B @ K mod q  (post-multiplication)

    Returns

    cipher_channel : (H, W) uint8 array (values in [0, q-1] ⊆ [0,255] for q≤256)
    
    """

    blocks, orig_shape, padded_shape = block_divide(channel, n)
    enc_blocks = []
    for b in blocks:
        enc = _encrypt_block(b, K, q, pre)
        if enc.max() >= q:
            print(f"WARNING: Block has values >= {q}: min={enc.min()}, max={enc.max()}")
        enc_blocks.append(enc)
    
    return block_merge(enc_blocks, orig_shape, padded_shape, n)


def decrypt_channel(
    cipher_channel: np.ndarray,
    K: np.ndarray,
    n: int,
    q: int,
    pre: bool = True,
) -> np.ndarray:
    """
    Reverse Hill Cipher on one (H, W) channel.

    Parameters

    cipher_channel : encrypted (H, W) uint8 / int array
    K              : (n, n) key matrix used during encryption (from SL_n(F_q))
    n              : block size
    q              : prime modulus
    pre            : must match the value used in encrypt_channel

    Returns

    plain_channel : (H, W) uint8 array
    """
    K_inv = _mod_inv_matrix(K, q)
    blocks, orig_shape, padded_shape = block_divide(cipher_channel, n)
    dec_blocks = [_decrypt_block(b, K_inv, q, pre) for b in blocks]
    return block_merge(dec_blocks, orig_shape, padded_shape, n)


# Quick self-test

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    Q = 251          # prime < 256 so pixel values stay in uint8 range
    N = 4            # 4×4 blocks  (matches the paper's illustration)

    print(f"Testing with n={N}, q={Q} ...")

    # key generation
    K = generate_key(N, Q, rng)
    print(f"Key K (det mod {Q} = {_mod_det(K, Q)}):\n{K}")
    K_inv = _mod_inv_matrix(K, Q)
    identity_check = _mat_mul_mod(K, K_inv, Q)
    assert np.all(identity_check == np.eye(N, dtype=np.int64)), \
        "K @ K_inv should be identity mod q"
    print("K @ K_inv ≡ I  ")

    # block divide / merge round-trip
    ch = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    blocks, orig, padded = block_divide(ch, N)
    restored = block_merge(blocks, orig, padded, N)
    assert np.array_equal(ch, restored), "block_divide/merge round-trip failed"
    print("block_divide → block_merge round-trip  ")

    # encrypt / decrypt round-trip (pre-multiplication)
    enc = encrypt_channel(ch, K, N, Q, pre=True)
    dec = decrypt_channel(enc, K, N, Q, pre=True)
    assert np.array_equal(ch, dec), "pre-multiply encrypt/decrypt round-trip failed"
    print("Pre-multiply encrypt → decrypt round-trip  ")

    # encrypt / decrypt round-trip (post-multiplication)
    enc2 = encrypt_channel(ch, K, N, Q, pre=False)
    dec2 = decrypt_channel(enc2, K, N, Q, pre=False)
    assert np.array_equal(ch, dec2), "post-multiply encrypt/decrypt round-trip failed"
    print("Post-multiply encrypt → decrypt round-trip  ")

    # non-square channel (padding test) 
    ch_rect = rng.integers(0, 256, size=(60, 70), dtype=np.uint8)
    enc_r = encrypt_channel(ch_rect, K, N, Q, pre=True)
    dec_r = decrypt_channel(enc_r, K, N, Q, pre=True)
    assert np.array_equal(ch_rect, dec_r), "rectangular channel round-trip failed"
    print("Rectangular (non-square) channel round-trip  ")

    print("\nAll self-tests passed.")
