"""Assemble the layered 3D proxy for a session."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schemas import LayerSpec, OriginCamera
from .mesh import MAX_HEIGHT, Mesh, build_shell, inflate_height, smooth


@dataclass
class ProxyLayer:
    name: str
    kind: str  # base | plane | inflate
    locked: bool
    mask: np.ndarray  # bool HxW, the pixels this layer owns in the texture
    mesh: Mesh


# Volume presets per proxy mode, used when no learned reconstructor is active.
# canonical_anime_head: uncapped dome, so a head is about as deep as it is wide.
# full_3d_reconstruction fallback: capped slab, safer for busts and full poses.
PRESETS = {
    "canonical_anime_head": dict(radius_cap=None, depth_scale=1.0),
    "full_3d_reconstruction": dict(radius_cap=0.28, depth_scale=0.85),
}


def build_layers(
    mask: np.ndarray,
    cam: OriginCamera,
    proxy_mode: str,
    specs: list[tuple[LayerSpec, np.ndarray]],
    learned_depth: np.ndarray | None = None,
) -> list[ProxyLayer]:
    """`specs` pairs each extra layer with its boolean mask (same size as `mask`).

    `learned_depth`, when a depth estimator is enabled, is relative depth in
    [0, 1] (1 = nearest). It replaces the inflated front surface and turns on
    discontinuity cutting, which is what keeps foreshortened limbs and held
    props separated in z from the torso.
    """
    h, _ = mask.shape
    preset = PRESETS[proxy_mode]
    sigma = h / 96.0  # rounds off the distance transform's medial ridges

    base_mask = mask.copy()
    for spec, layer_mask in specs:
        if spec.kind == "inflate":
            base_mask &= ~layer_mask

    def volume(m: np.ndarray) -> np.ndarray:
        return smooth(inflate_height(m, preset["radius_cap"], preset["depth_scale"]), m, sigma)

    base_h = volume(base_mask)
    discontinuity = None
    front = base_h
    if learned_depth is not None and base_mask.any():
        inside = learned_depth[base_mask]
        lo, hi = np.percentile(inside, [2, 98])
        rel = np.clip((learned_depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        # Keep the inflated rounding at the rim, take large-scale layout from the model.
        front = np.minimum(base_h + rel * MAX_HEIGHT * 0.8, MAX_HEIGHT) * base_mask
        discontinuity = 0.12

    layers = [
        ProxyLayer(
            name="base",
            kind="base",
            locked=False,
            mask=base_mask,
            mesh=build_shell("layer_base", base_mask, front, -base_h, cam, discontinuity=discontinuity),
        )
    ]
    for spec, layer_mask in specs:
        if not layer_mask.any():
            continue
        if spec.kind == "plane":
            # A flat shell just in front of whatever it sits on: glasses float
            # ahead of the eye sockets, so rotating shows frame-to-pupil parallax.
            under = front[layer_mask & base_mask]
            z0 = (float(np.percentile(under, 90)) if under.size else 0.0) + spec.offset
            z = np.full(mask.shape, min(z0, MAX_HEIGHT + 0.1), np.float32)
            mesh = build_shell(f"layer_{spec.name}", layer_mask, z, z - 0.01, cam)
        else:
            lh = volume(layer_mask)
            mesh = build_shell(f"layer_{spec.name}", layer_mask, lh + spec.offset, -lh + spec.offset, cam)
        layers.append(ProxyLayer(name=spec.name, kind=spec.kind, locked=spec.locked, mask=layer_mask, mesh=mesh))
    return layers


def layer_texture(projection: np.ndarray, layer: ProxyLayer, all_layers: list[ProxyLayer]) -> np.ndarray:
    """The projection texture restricted to one layer's pixels (anti-aliased)."""
    if layer.kind == "base" and len(all_layers) == 1:
        return projection
    m = cv2.GaussianBlur(layer.mask.astype(np.float32), (0, 0), 0.7)
    out = projection.copy()
    out[..., 3] = np.clip(out[..., 3].astype(np.float32) * m + 0.5, 0, 255).astype(np.uint8)
    return out
