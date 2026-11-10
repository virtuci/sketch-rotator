"""Per-pixel denoise strength for differential diffusion.

The viewport's mask buffer carries three signals per pixel:
  R  disoccluded (binary): never seen by the origin camera
  G  stretch = 1 - N.V_orig: how obliquely the origin camera saw the surface
  B  occluded-by-another-surface flag (subset of R, kept for diagnostics)
  A  coverage

Well-seen surfaces get almost no denoising, so badges, logos and hard cel
boundaries are moved by the geometric warp and are not redrawn. Strength
rises with stretch, and disoccluded regions get the full inpaint strength.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schemas import RefineOptions


@dataclass(frozen=True)
class DenoiseParams:
    visible_min: float
    visible_max: float
    disoccluded: float
    feather_px: float
    stretch_lo: float = 0.55  # N.V = 0.45: stretching starts to show
    stretch_hi: float = 0.85  # N.V = 0.15: texels smeared ~6x

    @classmethod
    def for_mode(cls, input_mode: str, opts: RefineOptions | None = None) -> "DenoiseParams":
        # Cel frames carry flat colour and emblems that must survive untouched;
        # lineart tolerates a little more to re-ink stretched strokes.
        base = dict(visible_min=0.05, visible_max=0.35, disoccluded=0.85, feather_px=3.0)
        if input_mode == "lineart":
            base["visible_min"] = 0.12
        if opts:
            for key, value in (
                ("visible_min", opts.denoise_visible_min),
                ("visible_max", opts.denoise_visible_max),
                ("disoccluded", opts.denoise_disoccluded),
                ("feather_px", opts.feather_px),
            ):
                if value is not None:
                    base[key] = value
        base["visible_max"] = max(base["visible_max"], base["visible_min"])
        return cls(**base)


def build_denoise_map(mask_rgba: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """float32 HxW strength in [0, 1]; 0 outside the object."""
    dis = mask_rgba[..., 0].astype(np.float32) / 255.0
    stretch = mask_rgba[..., 1].astype(np.float32) / 255.0
    cover = mask_rgba[..., 3].astype(np.float32) / 255.0

    t = np.clip((stretch - params.stretch_lo) / (params.stretch_hi - params.stretch_lo), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    visible = params.visible_min + (params.visible_max - params.visible_min) * t

    if params.feather_px > 0:
        # Grow the inpaint region slightly, then feather, so the seam between
        # warped and synthesised pixels falls inside the repainted area.
        r = max(1, int(round(params.feather_px)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        dis = cv2.GaussianBlur(cv2.dilate(dis, k), (0, 0), params.feather_px)
    strength = visible + (params.disoccluded - visible) * np.clip(dis, 0.0, 1.0)
    return (strength * (cover > 0.0)).astype(np.float32)
