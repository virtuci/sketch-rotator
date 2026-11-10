"""Background removal for the two production input modes.

Both functions return a `Prepared` bundle:

* `clean`      - the artist-facing RGBA layer (what gets pasted back).
* `projection` - the RGBA the viewport projects onto the proxy. Alpha is the
                 silhouette, and RGB is bled outward past the silhouette so
                 texture filtering and mipmaps never pull in background colour.
* `mask`       - boolean silhouette used to build the proxy geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from ..schemas import CHROMA_KEY_RGB


@dataclass
class Prepared:
    clean: np.ndarray
    projection: np.ndarray
    mask: np.ndarray
    warnings: list[str]


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def bleed_rgb(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Replace every pixel outside `known` with the colour of its nearest known pixel."""
    if known.all() or not known.any():
        return rgb
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


def drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


# --------------------------------------------------------------------------
# Mode A: monoline lineart on white or transparent
# --------------------------------------------------------------------------

def extract_ink(rgba: np.ndarray, lo: float = 0.06, hi: float = 0.85) -> np.ndarray:
    """Ink coverage in [0, 1]: 0 = paper, 1 = solid line.

    Works for both white-background and transparent-background sketches: the
    image is composited onto white first, so transparent pixels read as paper.
    `lo` removes paper tint and compression noise; `hi` makes dark grey solid.
    """
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    on_white = rgb * a + (1.0 - a)
    lum = on_white @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip(((1.0 - lum) - lo) / (hi - lo), 0.0, 1.0)


def _close_at_canvas_edges(lines: np.ndarray) -> np.ndarray:
    h, w = lines.shape
    k = max(2, max(h, w) // 100)
    out = lines.copy()
    for strip, edge in (
        (lines[:k].any(0), out[0]),
        (lines[-k:].any(0), out[-1]),
        (lines[:, :k].any(1), out[:, 0]),
        (lines[:, -k:].any(1), out[:, -1]),
    ):
        hits = np.flatnonzero(strip)
        if len(hits) and hits[-1] - hits[0] > 4 * k:
            edge[hits[0] : hits[-1] + 1] = 1
    return out


def lineart_silhouette(ink: np.ndarray) -> tuple[np.ndarray, bool]:
    """Region enclosed by the outline: everything the paper flood from the border cannot reach.

    Sketch outlines usually have small gaps, so the lines are closed with a
    growing kernel until the enclosed area is clearly larger than the ink
    itself. Returns (mask, closed_ok).

    A bust or waist-up drawing is cropped by the canvas, so its outline runs
    off an edge and is open there. Where strokes reach an image edge, that
    edge is treated as part of the outline between the outermost strokes.
    """
    h, w = ink.shape
    lines = (ink > 0.3).astype(np.uint8)
    lines = _close_at_canvas_edges(lines)
    ink_area = int(lines.sum())
    if ink_area == 0:
        return np.zeros((h, w), bool), False
    base = max(h, w)
    best = lines.astype(bool)
    for frac in (0.0, 0.004, 0.012, 0.03, 0.06):
        r = int(round(frac * base))
        pad = r + 2
        canvas = cv2.copyMakeBorder(lines, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        if r:
            canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, _disk(r))
        flood = canvas.copy()
        cv2.floodFill(flood, None, (0, 0), 2)
        inside = (flood != 2)[pad:-pad, pad:-pad]
        best = inside
        if inside.sum() >= 2.0 * ink_area:
            return inside, True
    return best, False


def preprocess_lineart(rgba: np.ndarray) -> Prepared:
    warnings: list[str] = []
    ink = extract_ink(rgba)
    mask, closed = lineart_silhouette(ink)
    if not closed:
        warnings.append(
            "The outline is too open to find an enclosed silhouette; the proxy covers the strokes only."
        )
        mask = cv2.dilate((ink > 0.3).astype(np.uint8), _disk(max(2, max(ink.shape) // 100))).astype(bool)
    h, w = ink.shape
    mask = drop_small_components(mask, max(16, (h * w) // 20000))
    # A line pixel is always part of the layer, even if a speck filter or the
    # flood disagreed, so no stroke is ever clipped by the silhouette.
    mask |= ink > 0.3

    alpha = np.clip(ink * 255.0 + 0.5, 0, 255).astype(np.uint8)
    clean = np.zeros((h, w, 4), np.uint8)
    clean[..., 3] = alpha  # pure black ink, alpha = coverage: no white halo possible

    # Projection texture: black ink over white paper inside the silhouette.
    paper = np.clip((1.0 - ink) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    sil = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7)
    projection = np.dstack([paper, paper, paper, np.clip(sil * 255.0 + 0.5, 0, 255).astype(np.uint8)])
    projection[..., 3][ink > 0.3] = 255
    return Prepared(clean=clean, projection=projection, mask=mask, warnings=warnings)


# --------------------------------------------------------------------------
# Mode B: cel-shaded keyframe on #229919
# --------------------------------------------------------------------------

def chroma_matte(
    rgba: np.ndarray,
    key: tuple[int, int, int] = CHROMA_KEY_RGB,
    tolerance: float = 8.0,
    band_px: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact key matting. Returns (rgb with spill removed, alpha in [0, 1]).

    Pixels within `tolerance` of the key are background. Pixels more than
    `band_px` from any background pixel are solid foreground and are left
    untouched, so flat cel colours stay bit-exact. In the thin band between,
    each pixel is an anti-aliased blend  p = a*F + (1-a)*K  of the key K and
    the foreground colour F of the nearest solid pixel, which gives alpha in
    closed form:  a = (p-K).(F-K) / |F-K|^2 . The band pixel then takes colour
    F, so no green survives, and because the matte is solved rather than
    eroded, the outer pixel of the black outline keeps its true coverage.

    The defaults assume a pixel-exact key and one pixel of anti-aliasing, which
    is what a renderer or paint program exports. For frames that were resampled
    or lossily compressed, raise `band_px` to the width of the soft edge and
    `tolerance` to the background noise level.
    """
    rgb = rgba[..., :3].astype(np.float32)
    k = np.array(key, np.float32)
    dist = np.linalg.norm(rgb - k, axis=-1)
    src_alpha = rgba[..., 3].astype(np.float32) / 255.0

    bg = (dist <= tolerance) | (src_alpha < 0.02)
    near_bg = cv2.dilate(bg.astype(np.uint8), np.ones((2 * band_px + 1,) * 2, np.uint8)).astype(bool)
    solid = ~near_bg
    band = near_bg & ~bg

    alpha = np.where(bg, 0.0, 1.0).astype(np.float32)
    out = rgb.copy()
    if band.any() and solid.any():
        d, idx = ndimage.distance_transform_edt(~solid, return_indices=True)
        fg = rgb[idx[0], idx[1]]
        pk = rgb - k
        fk = fg - k
        denom = np.maximum((fk * fk).sum(-1), 1.0)
        a = np.clip((pk * fk).sum(-1) / denom, 0.0, 1.0)
        # Features thinner than the band have no solid pixel nearby; borrowing
        # a far-away colour would be wrong, so those keep their own colour with
        # the key's green excess clamped, and take alpha from key distance.
        far = d > 3 * band_px + 1
        own = rgb.copy()
        own[..., 1] = np.minimum(own[..., 1], np.maximum(own[..., 0], own[..., 2]))
        a_far = np.clip(dist / float(np.linalg.norm(k)), 0.0, 1.0)
        alpha = np.where(band, np.where(far, a_far, a), alpha)
        out = np.where(band[..., None], np.where(far[..., None], own, fg), out)
    elif band.any():
        alpha = np.where(band, np.clip(dist / float(np.linalg.norm(k)), 0.0, 1.0), alpha)
        out[..., 1] = np.where(band, np.minimum(out[..., 1], np.maximum(out[..., 0], out[..., 2])), out[..., 1])
    alpha *= np.where(bg, 1.0, np.maximum(src_alpha, 0.0))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), alpha


def preprocess_cel_chroma(rgba: np.ndarray) -> Prepared:
    warnings: list[str] = []
    rgb, alpha = chroma_matte(rgba)
    if (alpha > 0.5).mean() > 0.98:
        warnings.append("No #229919 background found; the whole frame is treated as foreground.")
    if not (alpha > 0.5).any():
        warnings.append("The image is entirely #229919; there is nothing to rotate.")
    mask = alpha > 0.5
    a8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgb = bleed_rgb(rgb, alpha > 0.0)  # background pixels carry foreground colour, never green
    clean = np.dstack([rgb, a8])
    return Prepared(clean=clean, projection=clean.copy(), mask=mask, warnings=warnings)


def preprocess(rgba: np.ndarray, input_mode: str) -> Prepared:
    if input_mode == "lineart":
        return preprocess_lineart(rgba)
    if input_mode == "cel_chroma":
        return preprocess_cel_chroma(rgba)
    raise ValueError(f"unknown input_mode {input_mode!r}")
