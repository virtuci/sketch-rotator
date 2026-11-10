"""Proxy geometry built on the CPU from a silhouette and a height field.

World convention (see `OriginCamera`): image plane at z = 0 spanning
y in [-1, 1]; +z points at the origin camera. Every vertex is unprojected
along its own camera ray, so the mesh reprojects exactly onto the sketch when
viewed from the origin camera, at any origin focal length.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schemas import OriginCamera

MAX_HEIGHT = 0.6  # keeps the surface clear of the camera at 15mm (distance 1.25)


@dataclass
class Mesh:
    name: str
    vertices: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) uint32


def inflate_height(mask: np.ndarray, radius_cap: float | None, depth_scale: float = 1.0) -> np.ndarray:
    """Half-thickness of the volume under each pixel, in world units.

    Circular cross-section: at distance d from the silhouette the half
    thickness is sqrt(R^2 - (R - d)^2), with R the largest inscribed radius
    (optionally capped). A limb therefore becomes a tube as deep as it is
    wide, and a head a dome, without any learned prior.
    """
    h, _ = mask.shape
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) * (2.0 / h)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    radius = float(dt.max())
    if radius_cap is not None:
        radius = min(radius, radius_cap)
    radius = max(radius, 1e-4)
    d = np.minimum(dt, radius)
    height = np.sqrt(np.maximum(radius * radius - (radius - d) ** 2, 0.0)) * depth_scale
    return np.minimum(height, MAX_HEIGHT).astype(np.float32)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


def build_shell(
    name: str,
    mask: np.ndarray,
    front: np.ndarray,
    back: np.ndarray | None,
    cam: OriginCamera,
    max_cells: int = 256,
    discontinuity: float | None = None,
) -> Mesh:
    """Triangulate a height field over `mask` into a front (and optional back) surface.

    `front` / `back` are absolute z per pixel. The grid is grown by one cell
    past the silhouette so the texture alpha, not the triangle edge, decides
    the visible outline. With `discontinuity` set, cells whose corners differ
    by more than that in z are dropped instead of bridged, so a hand held
    toward the camera does not get welded to the chest by a wall of stretched
    triangles.
    """
    h, w = mask.shape
    scale = max_cells / max(h, w)
    nx = max(2, int(round(w * scale)))
    ny = max(2, int(round(h * scale)))
    m = cv2.resize(mask.astype(np.float32), (nx, ny), interpolation=cv2.INTER_AREA) > 0.0
    grown = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    def sample(field: np.ndarray) -> np.ndarray:
        # Average over the masked part of each cell.
        num = cv2.resize(field.astype(np.float32) * mask, (nx, ny), interpolation=cv2.INTER_AREA)
        den = cv2.resize(mask.astype(np.float32), (nx, ny), interpolation=cv2.INTER_AREA)
        return np.where(den > 0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)

    def with_rim(g: np.ndarray, rim: np.ndarray | None) -> np.ndarray:
        # Cells in the grown ring have no height of their own. For a closed
        # volume they sit on the mid-surface, where front and back meet, so
        # the two shells share one clean rim. For an open sheet they continue
        # the neighbouring surface.
        if rim is None:
            rim = cv2.dilate(g, np.ones((3, 3), np.uint8))
        return np.where(m, g, rim)

    px = (np.arange(nx) + 0.5) / nx * 2.0 - 1.0
    py = 1.0 - (np.arange(ny) + 0.5) / ny * 2.0
    gx, gy = np.meshgrid(px, py)

    quad = grown[:-1, :-1] & grown[:-1, 1:] & grown[1:, :-1] & grown[1:, 1:]
    index = np.arange(nx * ny).reshape(ny, nx)

    def surface(z: np.ndarray, flip: bool) -> tuple[np.ndarray, np.ndarray]:
        s = (cam.distance - z) / cam.distance
        verts = np.stack([gx * cam.aspect * s, gy * s, z], axis=-1).reshape(-1, 3)
        ok = quad
        if discontinuity is not None:
            corners = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
            ok = ok & ((corners.max(0) - corners.min(0)) < discontinuity)
        a, b = index[:-1, :-1][ok], index[:-1, 1:][ok]
        c, d = index[1:, :-1][ok], index[1:, 1:][ok]
        tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])  # CCW seen from +z
        if flip:
            tris = tris[:, ::-1]
        return verts, tris

    zf = sample(front)
    rim = None
    if back is not None:
        zb = sample(back)
        mid = (zf + zb) / 2.0
        # Nearest inside cell's mid-surface, pushed one cell outward.
        rim = np.where(m, mid, cv2.dilate(np.where(m, mid, -np.inf).astype(np.float32), np.ones((3, 3), np.uint8)))
        rim = np.where(np.isfinite(rim), rim, 0.0).astype(np.float32)
    verts, tris = surface(with_rim(zf, rim), flip=False)
    if back is not None:
        vb, tb = surface(with_rim(zb, rim), flip=True)
        tris = np.concatenate([tris, tb + len(verts)])
        verts = np.concatenate([verts, vb])

    # Drop unreferenced vertices.
    used = np.zeros(len(verts), bool)
    used[tris.ravel()] = True
    remap = np.cumsum(used) - 1
    verts = verts[used].astype(np.float32)
    tris = remap[tris].astype(np.uint32)
    return Mesh(name=name, vertices=verts, normals=vertex_normals(verts, tris), faces=tris)


def smooth(field: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Mask-aware blur: softens distance-transform ridges without leaking across the silhouette."""
    if sigma_px <= 0:
        return field
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(field * m, (0, 0), sigma_px)
    den = cv2.GaussianBlur(m, (0, 0), sigma_px)
    return np.where(mask, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def clay_render(height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lambert-shaded grey clay image of the inflated volume.

    This is the shaded stand-in that image-to-3D models need in place of flat
    lineart, and doubles as a debug view of what the proxy looks like.
    """
    h = height.shape[0]
    gy, gx = np.gradient(height * (h / 2.0))
    n = np.dstack([-gx, gy, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.35, 0.45, 0.82], np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.0, 1.0) * 0.75 + 0.2
    grey = np.where(mask, shade, 1.0)
    out = np.clip(grey * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([out, out, out, np.where(mask, 255, 0).astype(np.uint8)])


def register_to_silhouette(vertices: np.ndarray, mask: np.ndarray, cam: OriginCamera) -> np.ndarray:
    """Fit an externally generated mesh to the sketch.

    Image-to-3D models return geometry in their own normalised frame. This
    scales and translates it so its front-view bounding box matches the
    silhouette's bounding box on the image plane, and centres it in depth.
    It assumes the model's output already faces +z with +y up; meshes from a
    model with another convention must be re-oriented before this call.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return vertices
    h, w = mask.shape
    x0, x1 = xs.min() / w * 2 - 1, (xs.max() + 1) / w * 2 - 1
    y0, y1 = 1 - (ys.max() + 1) / h * 2, 1 - ys.min() / h * 2
    x0, x1 = x0 * cam.aspect, x1 * cam.aspect
    lo, hi = vertices.min(0), vertices.max(0)
    size = np.maximum(hi - lo, 1e-6)
    scale = min((x1 - x0) / size[0], (y1 - y0) / size[1])
    centre = (lo + hi) / 2.0
    out = (vertices - centre) * scale
    out[:, 0] += (x0 + x1) / 2.0
    out[:, 1] += (y0 + y1) / 2.0
    depth = out[:, 2].max()
    if depth > MAX_HEIGHT:
        out[:, 2] *= MAX_HEIGHT / depth
    return out.astype(np.float32)
