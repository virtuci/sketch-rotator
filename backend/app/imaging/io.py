"""Image I/O. Everything in the pipeline is straight-alpha RGBA uint8 (RGB order)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageError(ValueError):
    pass


def decode_rgba(data: bytes, max_side: int = 4096) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageError("not a decodable image (expected PNG or WebP)")
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    h, w = arr.shape[:2]
    if min(h, w) < 16:
        raise ImageError("image is too small")
    if max(h, w) > max_side:
        raise ImageError(f"image is larger than {max_side}px on its long side")
    return arr


def read_rgba(path: Path) -> np.ndarray:
    return decode_rgba(Path(path).read_bytes(), max_side=1 << 16)


def read_gray16(path: Path) -> np.ndarray:
    """Read a 16-bit (or 8-bit) grayscale PNG as float32 in [0, 1]."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageError(f"cannot read {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return arr.astype(np.float32) / scale


def write_rgba(path: Path, rgba: np.ndarray) -> None:
    code = cv2.COLOR_RGBA2BGRA if rgba.shape[2] == 4 else cv2.COLOR_RGB2BGR
    if not cv2.imwrite(str(path), cv2.cvtColor(rgba, code)):
        raise ImageError(f"cannot write {path}")


def write_gray(path: Path, gray: np.ndarray) -> None:
    if gray.dtype != np.uint8:
        gray = np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), gray):
        raise ImageError(f"cannot write {path}")
