"""
Load RGB images as NumPy tensors and pad so height and width are multiples of k
(key-matrix dimension) and equal (square tensor), so Arnold scrambling works on
each channel without extra reshaping.

Depends: numpy, Pillow (pip install numpy pillow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


class PaddedRGBImageLoader:
    """
    Read images from a directory, build a uint8 rank-3 tensor (N, N, 3).
    Each side is rounded up to a multiple of k; the shorter side is then padded
    to match the longer so H == W (square), compatible with the Arnold map.
    """

    def __init__(self, k: int, images_dir: PathLike | None = None) -> None:
        if k < 1:
            raise ValueError("k (key matrix dimension) must be a positive integer.")
        self.k = k
        base = Path(__file__).resolve().parent
        self.images_dir = Path(images_dir) if images_dir is not None else base / "Images"
        self.images_dir = self.images_dir.resolve()

        self._tensor: np.ndarray | None = None
        self._original_shape: tuple[int, int] | None = None
        self._padded_shape: tuple[int, int] | None = None

    def _pad_spatial(self, arr: np.ndarray) -> np.ndarray:
        """
        Pad (H, W, C) so the result is square with side length a multiple of k.

        First round H and W up to multiples of k; then pad the shorter axis
        (bottom/right) up to max of those two sides so Arnold can run on (N, N).
        """
        h, w = arr.shape[0], arr.shape[1]
        kh = ((h + self.k - 1) // self.k) * self.k
        kw = ((w + self.k - 1) // self.k) * self.k
        side = max(kh, kw)
        pad_h = side - h
        pad_w = side - w
        if pad_h == 0 and pad_w == 0:
            return arr
        return np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def load(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ``filename`` from ``images_dir``, return (R, G, B) each shape (N, N),
        uint8, with identical spatial padding on all channels (square, multiple of k).
        """
        path = self.images_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"No image at {path}")

        with Image.open(path) as im:
            im = im.convert("RGB")
            tensor = np.asarray(im, dtype=np.uint8)

        self._original_shape = (tensor.shape[0], tensor.shape[1])
        padded = self._pad_spatial(tensor)
        self._padded_shape = (padded.shape[0], padded.shape[1])
        self._tensor = padded

        r = padded[:, :, 0].copy()
        g = padded[:, :, 1].copy()
        b = padded[:, :, 2].copy()
        return r, g, b

    @property
    def tensor(self) -> np.ndarray:
        """Full padded rank-3 tensor (N, N, 3); call ``load`` first."""
        if self._tensor is None:
            raise RuntimeError("Call load() before accessing tensor.")
        return self._tensor

    @property
    def original_shape(self) -> tuple[int, int]:
        if self._original_shape is None:
            raise RuntimeError("Call load() before accessing original_shape.")
        return self._original_shape

    @property
    def padded_shape(self) -> tuple[int, int]:
        if self._padded_shape is None:
            raise RuntimeError("Call load() before accessing padded_shape.")
        return self._padded_shape


if __name__ == "__main__":
    import sys

    images_dir = Path(__file__).resolve().parent / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    demo_name = "_hill_demo.png"
    demo_path = images_dir / demo_name
    if not demo_path.is_file():
        # Small synthetic RGB image so __main__ runs without a user-provided file
        demo = np.zeros((5, 7, 3), dtype=np.uint8)
        demo[:, :, 0] = np.linspace(0, 255, 7, dtype=np.uint8)
        demo[:, :, 1] = 128
        demo[:, :, 2] = np.linspace(255, 0, 7, dtype=np.uint8)
        Image.fromarray(demo, mode="RGB").save(demo_path)

    k = 4
    loader = PaddedRGBImageLoader(k=k, images_dir=images_dir)
    R, G, B = loader.load(demo_name)

    print(f"k = {k}")
    print(f"images_dir = {images_dir}")
    print(f"original (H, W) = {loader.original_shape}")
    print(f"padded   (H, W) = {loader.padded_shape}")
    print(f"R, G, B shapes = {R.shape}, {G.shape}, {B.shape}")
    print(f"full tensor shape = {loader.tensor.shape}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(Install matplotlib for a visual check: pip install matplotlib)", file=sys.stderr)
        sys.exit(0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].imshow(Image.open(demo_path).convert("RGB"))
    axes[0, 0].set_title("Original (before pad)")
    axes[0, 1].imshow(loader.tensor)
    axes[0, 1].set_title(f"Padded square (multiples of k={k})")
    axes[1, 0].imshow(R, cmap="Reds", vmin=0, vmax=255)
    axes[1, 0].set_title("R channel (padded)")
    axes[1, 1].imshow(np.dstack([R, G, B]))
    axes[1, 1].set_title("Stacked RGB view")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
