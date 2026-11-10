"""Post-rotation refinement: sharpen warped pixels, fill disoccluded ones."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np

from ..config import Settings

Progress = Callable[[float, str], None]


@dataclass
class RefineInputs:
    warped: np.ndarray  # RGBA uint8, alpha = coverage
    depth: np.ndarray  # float32 [0, 1], 0 = near plane, 1 = far / background
    normal: np.ndarray  # RGB uint8, view-space normal * 0.5 + 0.5
    mask: np.ndarray  # RGBA uint8, see imaging.denoise_map
    strength: np.ndarray  # float32 [0, 1] differential-denoise map
    reference: np.ndarray  # RGBA uint8 original sketch, for style
    input_mode: str
    prompt: str | None = None
    seed: int | None = None
    steps: int | None = None


def classical_refine(inp: RefineInputs, progress: Progress) -> np.ndarray:
    """Model-free fallback. Returns RGB uint8.

    Visible pixels are kept exactly as warped. Disoccluded pixels cannot be
    drawn without a generative model, so they keep what is defensible from
    geometry alone:
      cel_chroma - the viewport's base fill, which is the heavily blurred
                   local colour of the drawing (hair stays hair-coloured,
                   skin stays skin-coloured), with no invented detail;
      lineart    - blank paper, plus an outline wherever the object's own
                   silhouette borders a disoccluded region (the proxy knows
                   where the form ends even though nothing was drawn there).
    """
    rgb = inp.warped[..., :3].copy()
    cover = inp.warped[..., 3] > 0
    hole = (inp.mask[..., 0] > 127) & cover
    progress(0.3, "filling disoccluded regions")
    if inp.input_mode == "lineart":
        rgb[hole] = 255
        width = max(2, round(max(rgb.shape[:2]) / 400))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
        rim = cover & ~cv2.erode(cover.astype(np.uint8), k).astype(bool)
        outline = rim & cv2.dilate(hole.astype(np.uint8), k).astype(bool)
        soft = cv2.GaussianBlur(outline.astype(np.float32), (0, 0), 0.6)
        rgb = np.clip(rgb.astype(np.float32) * (1.0 - soft[..., None]), 0, 255).astype(np.uint8)
    progress(1.0, "done")
    return rgb


@lru_cache(maxsize=1)
def _load_diffusion(settings: Settings, device: str):
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    nets = [ControlNetModel.from_pretrained(settings.controlnet_depth, torch_dtype=dtype)]
    if settings.controlnet_normal:
        nets.append(ControlNetModel.from_pretrained(settings.controlnet_normal, torch_dtype=dtype))
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        settings.sdxl_model, controlnet=nets, torch_dtype=dtype
    )
    if settings.ip_adapter_repo:
        pipe.load_ip_adapter(settings.ip_adapter_repo, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.7)
    return pipe.to(device), len(nets)


def diffusion_refine(inp: RefineInputs, progress: Progress, settings: Settings, device: str) -> np.ndarray:
    """SDXL img2img with depth (+ normal) ControlNet, IP-Adapter style reference
    and differential diffusion driven by `inp.strength`.

    Differential diffusion is done in the step callback. The run is scheduled
    for the strongest region. After each step, every latent whose strength is
    below the fraction of denoising still to come is reset to the original
    image noised to the current timestep, so a pixel with strength s is only
    free for the last s of the schedule. Pixels at strength ~0 therefore come
    out as the warped input, and the final image is additionally composited
    with the input wherever strength is below 0.02.

    Not yet validated against real weights; expect to tune scales and prompts.
    """
    import torch
    from PIL import Image

    pipe, n_nets = _load_diffusion(settings, device)
    h, w = inp.warped.shape[:2]
    # SDXL wants multiples of 8 and about one megapixel.
    scale = min(1.0, (1024 * 1024 / (h * w)) ** 0.5)
    gw, gh = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)

    bg = np.array([255, 255, 255] if inp.input_mode == "lineart" else [128, 128, 128], np.float32)
    a = inp.warped[..., 3:4].astype(np.float32) / 255.0
    init = np.clip(inp.warped[..., :3] * a + bg * (1 - a), 0, 255).astype(np.uint8)
    init_img = Image.fromarray(init).resize((gw, gh), Image.LANCZOS)
    depth_img = Image.fromarray(np.clip((1.0 - inp.depth) * 255, 0, 255).astype(np.uint8)).convert("RGB").resize((gw, gh))
    control = [depth_img] + ([Image.fromarray(inp.normal).resize((gw, gh))] if n_nets > 1 else [])
    ref = inp.reference[..., :3] if inp.input_mode == "cel_chroma" else 255 - np.repeat(inp.reference[..., 3:4], 3, 2)

    max_strength = float(max(inp.strength.max(), 0.05))
    steps = inp.steps or 28
    strength_lat = torch.from_numpy(cv2.resize(inp.strength, (gw // 8, gh // 8), interpolation=cv2.INTER_AREA))
    strength_lat = strength_lat.to(device)[None, None]
    generator = torch.Generator(device).manual_seed(inp.seed) if inp.seed is not None else None

    with torch.no_grad():
        px = pipe.image_processor.preprocess(init_img).to(device, pipe.vae.dtype)
        clean = pipe.vae.encode(px).latent_dist.mode() * pipe.vae.config.scaling_factor
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)

    def on_step(p, i, t, kwargs):
        timesteps = p.scheduler.timesteps
        remaining = (len(timesteps) - 1 - i) / max(len(timesteps), 1) * max_strength
        if i + 1 < len(timesteps):
            keep = (strength_lat < remaining).to(clean.dtype)
            noised = p.scheduler.add_noise(clean, noise, timesteps[i + 1 : i + 2])
            kwargs["latents"] = kwargs["latents"] * (1 - keep) + noised.to(clean.dtype) * keep
        progress((i + 1) / len(timesteps), f"denoising step {i + 1}/{len(timesteps)}")
        return kwargs

    style = "monochrome, clean lineart, white background" if inp.input_mode == "lineart" else "anime screencap, flat color, cel shading, hard shadows"
    result = pipe(
        prompt=inp.prompt or f"{style}, same character, consistent line weight",
        negative_prompt="blurry, gradient shading, 3d render, sketchy, extra limbs, text artifacts",
        image=init_img,
        control_image=control,
        ip_adapter_image=Image.fromarray(ref) if settings.ip_adapter_repo else None,
        strength=max_strength,
        num_inference_steps=int(round(steps / max_strength)),  # img2img runs steps * strength of these
        controlnet_conditioning_scale=[0.9, 0.6][:n_nets],
        guidance_scale=5.0,
        generator=generator,
        callback_on_step_end=on_step,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    out = np.asarray(result.resize((w, h), Image.LANCZOS), np.uint8)
    protect = (inp.strength < 0.02)[..., None]
    return np.where(protect, init, out)


def refine(inp: RefineInputs, progress: Progress, settings: Settings, device: str) -> tuple[np.ndarray, str]:
    """Returns (RGB uint8, name of the refiner that ran)."""
    if settings.refiner == "diffusion":
        return diffusion_refine(inp, progress, settings, device), "diffusion"
    return classical_refine(inp, progress), "classical"
