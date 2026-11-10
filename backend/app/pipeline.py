"""The two GPU jobs: build a session's 3D proxy, and refine a rotated capture.

Both are plain synchronous functions of (payload, progress, device) with
JSON-serialisable payloads and results, so the same code runs in the gateway's
thread pool (inline queue) and inside a Celery GPU worker. Large data moves
through the shared session directory, never through the queue.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .config import Settings, get_settings
from .imaging import io
from .imaging.denoise_map import DenoiseParams, build_denoise_map
from .imaging.postprocess import postprocess
from .imaging.preprocess import preprocess
from .models import hooks
from .models.refiner import RefineInputs, refine
from .proxy.builder import ProxyLayer, build_layers, layer_texture
from .proxy.glb import write_glb
from .proxy.mesh import Mesh, clay_render, inflate_height, register_to_silhouette, vertex_normals
from .schemas import (
    CameraMeta,
    LayerInfo,
    LayerSpec,
    OriginCamera,
    RefineOptions,
    RefineResult,
    SessionInitResponse,
)
from .storage import Storage

Progress = Callable[[str, float, str], None]  # stage, fraction of whole job, message


class PipelineError(ValueError):
    """A problem with the request that the artist can act on."""


def _layer_masks(
    specs: list[LayerSpec],
    sdir,
    shape: tuple[int, int],
    silhouette: np.ndarray,
    rgb: np.ndarray,
    settings: Settings,
    device: str,
    warnings: list[str],
) -> list[tuple[LayerSpec, np.ndarray]]:
    out = []
    seen = {"base"}
    for spec in specs:
        if spec.name in seen:
            raise PipelineError(f"duplicate layer name {spec.name!r}")
        seen.add(spec.name)
        if spec.mask_index is not None:
            path = sdir / f"layer_mask_{spec.mask_index}.png"
            if not path.exists():
                raise PipelineError(f"layer {spec.name!r} refers to a mask that was not uploaded")
            m = io.read_rgba(path)
            if m.shape[:2] != shape:
                raise PipelineError(f"mask for layer {spec.name!r} must be the same size as the sketch")
            # Accept either a painted-on-transparent mask or white-on-black.
            opaque = m[..., 3].min() == 255
            mask = (m[..., :3].max(-1) > 127) if opaque else (m[..., 3] > 127)
        elif spec.points or spec.box:
            seg = hooks.segmenter(settings, device)
            if seg is None:
                warnings.append(
                    f"Layer {spec.name!r} needs SAM-2 (TS_SEGMENTER=sam2), which is not enabled; "
                    "upload a painted mask for it instead."
                )
                continue
            mask = seg(rgb, spec)
        else:
            raise PipelineError(f"layer {spec.name!r} has neither a mask nor prompts")
        mask = mask & silhouette
        if not mask.any():
            warnings.append(f"Layer {spec.name!r} does not overlap the sketch and was skipped.")
            continue
        out.append((spec, mask))
    return out


def init_session(payload: dict, progress: Progress, device: str = "cpu") -> dict:
    settings = get_settings()
    storage = Storage(settings.data_dir)
    sid = payload["session_id"]
    sdir = storage.dir(sid)
    input_mode, proxy_mode = payload["input_mode"], payload["proxy_mode"]
    specs = [LayerSpec(**s) for s in payload.get("layers", [])]

    progress("preprocess", 0.05, "removing background")
    rgba = io.read_rgba(sdir / "input.png")
    h, w = rgba.shape[:2]
    prep = preprocess(rgba, input_mode)
    warnings = list(prep.warnings)
    if not prep.mask.any():
        raise PipelineError("Nothing was left after background removal. Check the input mode.")
    io.write_rgba(sdir / "clean.png", prep.clean)

    cam = OriginCamera.for_image(w, h, payload.get("origin_focal_mm", 85.0))
    opaque_rgb = prep.projection[..., :3]

    if payload.get("separate_props") and not specs:
        warnings.append("Prop separation was requested but no layer masks or prompts were supplied.")
    progress("layers", 0.2, "splitting layers")
    layer_specs = _layer_masks(specs, sdir, (h, w), prep.mask, opaque_rgb, settings, device, warnings)

    preset_height = inflate_height(prep.mask, 0.28)
    clay = clay_render(preset_height, prep.mask)
    io.write_rgba(sdir / "clay.png", clay)

    progress("proxy", 0.35, "building 3D proxy")
    proxy_source = "inflate"
    learned_depth = None
    depth_fn = hooks.depth_estimator(settings, device)
    if depth_fn is not None:
        # Flat lineart has no shading for a depth model to read, so Mode A is
        # estimated from the clay render; cel frames are estimated directly.
        learned_depth = depth_fn(clay[..., :3] if input_mode == "lineart" else opaque_rgb)
        learned_depth = cv2.resize(learned_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        proxy_source = "inflate+depth_anything"

    layers = build_layers(prep.mask, cam, proxy_mode, layer_specs, learned_depth)

    recon = hooks.reconstructor(settings, device) if proxy_mode == "full_3d_reconstruction" else None
    if recon is not None:
        progress("proxy", 0.5, "reconstructing mesh")
        verts, faces = recon(clay)
        verts = register_to_silhouette(verts, layers[0].mask, cam)
        layers[0] = ProxyLayer(
            name="base", kind="base", locked=False, mask=layers[0].mask,
            mesh=Mesh("layer_base", verts, vertex_normals(verts, faces), faces),
        )
        proxy_source = "hunyuan3d"
    elif proxy_mode == "canonical_anime_head":
        warnings.append(
            "No fitted head base mesh is installed; the head proxy is an inflated dome of the silhouette."
        )

    progress("export", 0.85, "writing mesh and textures")
    write_glb(sdir / "proxy.glb", [layer.mesh for layer in layers])
    infos = []
    for layer in layers:
        io.write_rgba(sdir / f"tex_{layer.name}.png", layer_texture(prep.projection, layer, layers))
        infos.append(
            LayerInfo(
                name=layer.name,
                node=layer.mesh.name,
                texture_url=storage.url(sid, f"tex_{layer.name}.png"),
                locked=layer.locked,
                kind=layer.kind,
            )
        )

    response = SessionInitResponse(
        session_id=sid,
        input_mode=input_mode,
        proxy_mode=proxy_mode,
        width=w,
        height=h,
        texture_url=storage.url(sid, "clean.png"),
        mesh_url=storage.url(sid, "proxy.glb"),
        layers=infos,
        origin_camera=cam,
        proxy_source=proxy_source,
        warnings=warnings,
    )
    storage.write_meta(sid, response.model_dump())
    progress("done", 1.0, "proxy ready")
    return response.model_dump()


def refine_rotation(payload: dict, progress: Progress, device: str = "cpu") -> dict:
    settings = get_settings()
    storage = Storage(settings.data_dir)
    sid, job_id = payload["session_id"], payload["job_id"]
    meta = storage.read_meta(sid)
    input_mode = meta["input_mode"]
    camera = CameraMeta(**payload["camera"])
    options = RefineOptions(**payload.get("options", {}))
    jdir = storage.dir(sid) / "refine" / job_id

    progress("load", 0.05, "reading viewport buffers")
    warped = io.read_rgba(jdir / "projected_rgba.png")
    mask = io.read_rgba(jdir / "disocclusion_mask.png")
    normal = io.read_rgba(jdir / "viewport_normal.png")[..., :3]
    depth = io.read_gray16(jdir / "viewport_depth.png")
    shape = (camera.height, camera.width)
    for name, arr in (("projected_rgba", warped), ("disocclusion_mask", mask), ("viewport_normal", normal), ("viewport_depth", depth)):
        if arr.shape[:2] != shape:
            raise PipelineError(f"{name} is {arr.shape[1]}x{arr.shape[0]}, camera metadata says {camera.width}x{camera.height}")
    if not (warped[..., 3] > 0).any():
        raise PipelineError("The capture is empty: the object is entirely out of frame.")

    progress("mask", 0.1, "building denoise map")
    strength = build_denoise_map(mask, DenoiseParams.for_mode(input_mode, options))
    io.write_gray(jdir / "denoise_map.png", strength)

    def step(fraction: float, message: str) -> None:
        progress("refine", 0.15 + 0.7 * fraction, message)

    refined, refiner_name = refine(
        RefineInputs(
            warped=warped, depth=depth, normal=normal, mask=mask, strength=strength,
            reference=io.read_rgba(storage.dir(sid) / "clean.png"),
            input_mode=input_mode, prompt=options.prompt, seed=options.seed, steps=options.steps,
        ),
        step, settings, device,
    )

    progress("postprocess", 0.9, "cleaning lines" if input_mode == "lineart" else "compositing on #229919")
    layer, flat = postprocess(refined, warped[..., 3], input_mode)
    io.write_rgba(jdir / "result_rgba.png", layer)
    io.write_rgba(jdir / "result_flat.png", flat)

    result = RefineResult(
        result_rgba_url=storage.url(sid, "refine", job_id, "result_rgba.png"),
        result_flat_url=storage.url(sid, "refine", job_id, "result_flat.png"),
        denoise_map_url=storage.url(sid, "refine", job_id, "denoise_map.png"),
        refiner=refiner_name,
        width=camera.width,
        height=camera.height,
    )
    progress("done", 1.0, "rotation refined")
    return result.model_dump()


TASKS: dict[str, Callable[[dict, Progress, str], dict]] = {
    "init_session": init_session,
    "refine_rotation": refine_rotation,
}
