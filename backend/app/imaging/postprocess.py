"""Turn the refiner's output into the layer the artist pastes back."""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas import CHROMA_KEY_RGB
from .preprocess import bleed_rgb, drop_small_components, extract_ink


def postprocess_lineart(refined_rgb: np.ndarray, coverage: np.ndarray, speck_area: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Crisp black ink on transparent, plus the same on white.

    The ink is pushed through a steep contrast curve around 50% coverage. That
    removes grey haze from warping and diffusion and restores a hard stroke
    edge while keeping about one pixel of anti-aliasing (a pure threshold
    would produce stair-stepped lines).
    """
    rgba = np.dstack([refined_rgb, np.full(refined_rgb.shape[:2], 255, np.uint8)])
    ink = extract_ink(rgba, lo=0.0, hi=1.0) * (coverage > 0)
    ink = np.clip((ink - 0.5) * 3.0 + 0.5, 0.0, 1.0)
    ink *= drop_small_components(ink > 0.25, speck_area)
    alpha = np.clip(ink * 255.0 + 0.5, 0, 255).astype(np.uint8)
    layer = np.zeros((*alpha.shape, 4), np.uint8)
    layer[..., 3] = alpha
    white = np.repeat((255 - alpha)[..., None], 3, axis=2)
    return layer, white


def postprocess_cel_chroma(
    refined_rgb: np.ndarray, coverage: np.ndarray, key: tuple[int, int, int] = CHROMA_KEY_RGB
) -> tuple[np.ndarray, np.ndarray]:
    """RGBA layer plus a composite on exact #229919.

    `coverage` is the anti-aliased alpha from the viewport. Edge pixels take
    the colour of the nearest fully covered pixel before compositing, so the
    only green in the frame is the key itself, and every pixel outside the
    matte is exactly the key value.
    """
    alpha = coverage.astype(np.float32) / 255.0
    rgb = bleed_rgb(refined_rgb, coverage >= 250) if (coverage >= 250).any() else refined_rgb
    layer = np.dstack([rgb, coverage])
    k = np.array(key, np.float32)
    flat = rgb.astype(np.float32) * alpha[..., None] + k * (1.0 - alpha[..., None])
    flat = np.clip(flat + 0.5, 0, 255).astype(np.uint8)
    flat[coverage == 0] = key
    return layer, flat


def postprocess(refined_rgb: np.ndarray, coverage: np.ndarray, input_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if input_mode == "lineart":
        return postprocess_lineart(refined_rgb, coverage)
    if input_mode == "cel_chroma":
        return postprocess_cel_chroma(refined_rgb, coverage)
    raise ValueError(f"unknown input_mode {input_mode!r}")
