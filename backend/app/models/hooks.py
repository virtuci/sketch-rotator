"""Optional learned models, loaded lazily on the GPU worker.

Each hook is selected by a `TS_*` setting and returns None when it is switched
off, in which case the pipeline uses its CPU fallback and says so in the
response. Model imports live inside the loaders so the gateway and the tests
never need torch.

None of the loaders in this file has been run against real weights in this
repository's CI; treat them as integration points to validate on the GPU box.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

from ..config import Settings
from ..schemas import LayerSpec

DepthFn = Callable[[np.ndarray], np.ndarray]  # RGB uint8 -> float32 HxW, 1 = nearest
SegmentFn = Callable[[np.ndarray, LayerSpec], np.ndarray]  # RGB uint8, prompts -> bool HxW
ReconstructFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]  # RGBA clay -> (vertices, faces)


@lru_cache(maxsize=4)
def load_depth_estimator(model_id: str, device: str) -> DepthFn:
    """Depth-Anything-V2 through the transformers depth-estimation pipeline.

    Point TS_DEPTH_MODEL at an anime-tuned checkpoint for flat lineart; the
    pipeline feeds it the clay render, not the raw lines, for Mode A.
    """
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model_id, device=device)

    def run(rgb: np.ndarray) -> np.ndarray:
        depth = np.asarray(pipe(Image.fromarray(rgb))["depth"], np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        return (depth - lo) / max(hi - lo, 1e-6)

    return run


@lru_cache(maxsize=2)
def load_segmenter(model_id: str, device: str) -> SegmentFn:
    """SAM-2 image predictor driven by the artist's point / box prompts."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def run(rgb: np.ndarray, spec: LayerSpec) -> np.ndarray:
        predictor.set_image(rgb)
        points = np.array(spec.points, np.float32) if spec.points else None
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=np.ones(len(spec.points), np.int32) if spec.points else None,
            box=np.array(spec.box, np.float32) if spec.box else None,
            multimask_output=True,
        )
        return masks[int(np.argmax(scores))].astype(bool)

    return run


@lru_cache(maxsize=1)
def load_reconstructor(model_id: str, device: str) -> ReconstructFn:
    """Hunyuan3D-2 shape generation from the shaded clay proxy."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from PIL import Image

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)
    pipe.to(device)

    def run(clay_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mesh = pipe(image=Image.fromarray(clay_rgba))[0]
        return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32)

    return run


def depth_estimator(settings: Settings, device: str) -> DepthFn | None:
    if settings.depth_estimator != "depth_anything":
        return None
    return load_depth_estimator(settings.depth_model, device)


def segmenter(settings: Settings, device: str) -> SegmentFn | None:
    if settings.segmenter != "sam2":
        return None
    return load_segmenter(settings.sam2_model, device)


def reconstructor(settings: Settings, device: str) -> ReconstructFn | None:
    if settings.reconstructor != "hunyuan3d":
        return None
    return load_reconstructor(settings.hunyuan_model, device)
