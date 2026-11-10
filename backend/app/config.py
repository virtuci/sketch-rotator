"""Runtime configuration, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("TS_DATA_DIR", "./data")).resolve())
    # "inline": jobs run in a thread pool inside the gateway (dev / single box).
    # "celery": jobs go to GPU workers through Redis.
    queue_backend: str = field(default_factory=lambda: os.environ.get("TS_QUEUE_BACKEND", "inline"))
    redis_url: str = field(default_factory=lambda: os.environ.get("TS_REDIS_URL", "redis://localhost:6379/0"))
    # Inline queue only: one worker thread per entry, each pinned to that device.
    devices: tuple[str, ...] = field(
        default_factory=lambda: tuple(d for d in os.environ.get("TS_DEVICES", "cpu").split(",") if d)
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o for o in os.environ.get("TS_CORS_ORIGINS", "http://localhost:3000").split(",") if o
        )
    )
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("TS_MAX_UPLOAD_MB", "64")) * 2**20)
    max_image_side: int = field(default_factory=lambda: int(os.environ.get("TS_MAX_IMAGE_SIDE", "4096")))

    # Model switches. Everything defaults to the CPU fallbacks so the stack runs
    # end to end without weights; flip these on the GPU workers.
    refiner: str = field(default_factory=lambda: os.environ.get("TS_REFINER", "classical"))  # classical | diffusion
    reconstructor: str = field(default_factory=lambda: os.environ.get("TS_RECONSTRUCTOR", "inflate"))  # inflate | hunyuan3d
    depth_estimator: str = field(default_factory=lambda: os.environ.get("TS_DEPTH_ESTIMATOR", "none"))  # none | depth_anything
    segmenter: str = field(default_factory=lambda: os.environ.get("TS_SEGMENTER", "none"))  # none | sam2

    sdxl_model: str = field(default_factory=lambda: os.environ.get("TS_SDXL_MODEL", "cagliostrolab/animagine-xl-3.1"))
    controlnet_depth: str = field(
        default_factory=lambda: os.environ.get("TS_CONTROLNET_DEPTH", "diffusers/controlnet-depth-sdxl-1.0")
    )
    controlnet_normal: str = field(default_factory=lambda: os.environ.get("TS_CONTROLNET_NORMAL", ""))
    ip_adapter_repo: str = field(default_factory=lambda: os.environ.get("TS_IP_ADAPTER_REPO", "h94/IP-Adapter"))
    depth_model: str = field(
        default_factory=lambda: os.environ.get("TS_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")
    )
    hunyuan_model: str = field(default_factory=lambda: os.environ.get("TS_HUNYUAN_MODEL", "tencent/Hunyuan3D-2"))
    sam2_model: str = field(default_factory=lambda: os.environ.get("TS_SAM2_MODEL", "facebook/sam2.1-hiera-large"))


def get_settings() -> Settings:
    return Settings()
