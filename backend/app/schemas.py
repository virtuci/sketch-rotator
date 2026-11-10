"""Request/response contracts shared by the gateway, the workers and bridge plugins."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

InputMode = Literal["lineart", "cel_chroma"]
ProxyMode = Literal["canonical_anime_head", "full_3d_reconstruction"]

CHROMA_KEY_RGB = (0x22, 0x99, 0x19)  # #229919
SENSOR_HEIGHT_MM = 24.0  # full-frame; focal lengths in the UI are 35mm-equivalent


def focal_to_fov_y(focal_mm: float) -> float:
    """Vertical field of view in radians for a 35mm-equivalent focal length."""
    return 2.0 * math.atan(SENSOR_HEIGHT_MM / (2.0 * focal_mm))


class OriginCamera(BaseModel):
    """The camera the sketch is assumed to have been drawn from.

    World convention (shared with the frontend): the image plane sits at z = 0
    and spans y in [-1, 1] and x in [-aspect, aspect]. The camera is on +Z
    looking down -Z, at the distance where that plane exactly fills the frame.
    """

    focal_mm: float
    fov_y: float
    aspect: float
    distance: float
    near: float
    far: float

    @classmethod
    def for_image(cls, width: int, height: int, focal_mm: float) -> "OriginCamera":
        fov = focal_to_fov_y(focal_mm)
        dist = 1.0 / math.tan(fov / 2.0)
        return cls(
            focal_mm=focal_mm,
            fov_y=fov,
            aspect=width / height,
            distance=dist,
            near=max(dist - 2.5, dist * 0.02),
            far=dist + 2.5,
        )


class LayerSpec(BaseModel):
    """An extra layer the artist (or SAM-2) carved out of the sketch."""

    name: str = Field(pattern=r"^[a-zA-Z0-9_\-]{1,32}$")
    # plane: thin shell floating `offset` in front of the base surface (glasses).
    # inflate: its own volume, removed from the base layer (desk, chair, prop).
    kind: Literal["plane", "inflate"] = "plane"
    offset: float = Field(default=0.06, ge=0.0, le=0.5)
    locked: bool = False
    mask_index: int | None = None  # index into the uploaded `layer_masks`
    # SAM-2 prompts in source-image pixels, used when no mask is uploaded.
    points: list[tuple[float, float]] = []
    box: tuple[float, float, float, float] | None = None


class LayerInfo(BaseModel):
    name: str
    node: str  # node name inside the proxy .glb
    texture_url: str
    locked: bool
    kind: Literal["base", "plane", "inflate"]


class SessionInitResponse(BaseModel):
    session_id: str
    input_mode: InputMode
    proxy_mode: ProxyMode
    width: int
    height: int
    texture_url: str  # clean RGBA of the whole sketch
    mesh_url: str
    layers: list[LayerInfo]
    origin_camera: OriginCamera
    proxy_source: str  # which generator actually produced the mesh
    warnings: list[str] = []


class CameraMeta(BaseModel):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    focal_mm: float
    fov_y: float
    distance: float
    near: float
    far: float
    width: int
    height: int


class RefineOptions(BaseModel):
    """Per-request overrides for the differential-denoise map."""

    denoise_visible_min: float | None = Field(default=None, ge=0.0, le=1.0)
    denoise_visible_max: float | None = Field(default=None, ge=0.0, le=1.0)
    denoise_disoccluded: float | None = Field(default=None, ge=0.0, le=1.0)
    feather_px: float | None = Field(default=None, ge=0.0, le=64.0)
    seed: int | None = None
    prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)


class RefineAccepted(BaseModel):
    job_id: str
    ws_url: str
    status_url: str


class RefineResult(BaseModel):
    result_rgba_url: str  # transparent layer, ready for the clipboard
    result_flat_url: str  # on white (lineart) or on #229919 (cel_chroma)
    denoise_map_url: str
    refiner: str
    width: int
    height: int
