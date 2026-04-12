"""
Discrete Arnold (cat) map for scrambling square 2D channels, and the inverse map.

Typical use: after per-channel Hill encryption, permute pixels with ``scramble``;
use ``inverse`` before decryption. Requires height == width (classical Arnold on Z_N^2).

Depends: numpy.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

ChannelBundle = Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


class ArnoldScrambler:
    """
    Forward and inverse Arnold cat map on square matrices.

    One iteration moves pixel (i, j) to ((i + j) mod N, (i + 2j) mod N)
    with N = height = width (row/column indices in [0, N)).
    """

    @staticmethod
    def _assert_square(channel: np.ndarray) -> int:
        if channel.ndim != 2:
            raise ValueError("Each channel must be a 2D array.")
        h, w = channel.shape
        if h != w:
            raise ValueError(
                "Arnold transform requires a square channel (H == W); "
                f"got shape {channel.shape}."
            )
        return h

    @staticmethod
    def _one_step_forward(channel: np.ndarray, n: int) -> np.ndarray:
        rows, cols = np.indices((n, n))
        i_new = (rows + cols) % n
        j_new = (rows + 2 * cols) % n
        out = np.empty_like(channel)
        out[i_new, j_new] = channel[rows, cols]
        return out

    @staticmethod
    def _one_step_inverse(channel: np.ndarray, n: int) -> np.ndarray:
        rows, cols = np.indices((n, n))
        i_orig = (2 * rows - cols) % n
        j_orig = (cols - rows) % n
        out = np.empty_like(channel)
        out[i_orig, j_orig] = channel[rows, cols]
        return out

    @classmethod
    def scramble(
        cls,
        channels: ChannelBundle,
        iterations: int = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Apply the Arnold map ``iterations`` times.

        ``channels`` may be:
        - a single (N, N) array;
        - a (N, N, 3) array (RGB planes scrambled independently, same iteration count);
        - a tuple (R, G, B) of (N, N) arrays.
        """
        if iterations < 0:
            raise ValueError("iterations must be non-negative.")

        if isinstance(channels, tuple) and len(channels) == 3:
            r, g, b = channels
            return (
                cls._scramble_single(r, iterations),
                cls._scramble_single(g, iterations),
                cls._scramble_single(b, iterations),
            )

        arr = np.asarray(channels)
        if arr.ndim == 2:
            return cls._scramble_single(arr, iterations)
        if arr.ndim == 3 and arr.shape[2] == 3:
            out = np.empty_like(arr)
            for c in range(3):
                out[:, :, c] = cls._scramble_single(arr[:, :, c], iterations)
            return out

        raise ValueError(
            "Expected a (N,N) array, a (N,N,3) array, or a tuple of three (N,N) arrays."
        )

    @classmethod
    def inverse(
        cls,
        channels: ChannelBundle,
        iterations: int = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Apply the inverse Arnold map ``iterations`` times (undo ``scramble``)."""
        if iterations < 0:
            raise ValueError("iterations must be non-negative.")

        if isinstance(channels, tuple) and len(channels) == 3:
            r, g, b = channels
            return (
                cls._inverse_single(r, iterations),
                cls._inverse_single(g, iterations),
                cls._inverse_single(b, iterations),
            )

        arr = np.asarray(channels)
        if arr.ndim == 2:
            return cls._inverse_single(arr, iterations)
        if arr.ndim == 3 and arr.shape[2] == 3:
            out = np.empty_like(arr)
            for c in range(3):
                out[:, :, c] = cls._inverse_single(arr[:, :, c], iterations)
            return out

        raise ValueError(
            "Expected a (N,N) array, a (N,N,3) array, or a tuple of three (N,N) arrays."
        )

    @classmethod
    def _scramble_single(cls, channel: np.ndarray, iterations: int) -> np.ndarray:
        n = cls._assert_square(channel)
        out = np.array(channel, copy=True, order="C")
        for _ in range(iterations):
            out = cls._one_step_forward(out, n)
        return out

    @classmethod
    def _inverse_single(cls, channel: np.ndarray, iterations: int) -> np.ndarray:
        n = cls._assert_square(channel)
        out = np.array(channel, copy=True, order="C")
        for _ in range(iterations):
            out = cls._one_step_inverse(out, n)
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 64
    base = rng.integers(0, 256, size=(n, n), dtype=np.uint8)

    iters = 7
    scrambled = ArnoldScrambler.scramble(base, iterations=iters)
    restored = ArnoldScrambler.inverse(scrambled, iterations=iters)
    assert np.array_equal(base, restored)

    stack = rng.integers(0, 256, size=(n, n, 3), dtype=np.uint8)
    s3 = ArnoldScrambler.scramble(stack, iterations=iters)
    r3 = ArnoldScrambler.inverse(s3, iterations=iters)
    assert np.array_equal(stack, r3)

    r, g, b = stack[:, :, 0], stack[:, :, 1], stack[:, :, 2]
    rt, gt, bt = ArnoldScrambler.scramble((r, g, b), iterations=iters)
    assert np.array_equal(np.stack([rt, gt, bt], axis=2), s3)

    print("Arnold round-trip checks passed (1-channel, (N,N,3), and RGB tuple).")
