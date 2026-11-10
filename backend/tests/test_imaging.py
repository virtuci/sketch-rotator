import cv2
import numpy as np

from app.imaging.denoise_map import DenoiseParams, build_denoise_map
from app.imaging.postprocess import postprocess_cel_chroma, postprocess_lineart
from app.imaging.preprocess import chroma_matte, preprocess_cel_chroma, preprocess_lineart
from app.schemas import RefineOptions

from .conftest import BADGE, KEY, SKIN


def test_lineart_becomes_black_ink_with_alpha(lineart_png):
    prep = preprocess_lineart(lineart_png)
    assert not prep.warnings
    assert prep.clean[..., :3].max() == 0  # pure black: nothing white left to halo
    alpha = prep.clean[..., 3]
    assert alpha[128, 128] == 0  # paper inside the head is transparent
    assert alpha.max() == 255
    assert 0 < (alpha > 0).mean() < 0.12


def test_lineart_silhouette_is_the_enclosed_region(lineart_png):
    prep = preprocess_lineart(lineart_png)
    assert prep.mask[128, 128]  # interior counts as the object
    assert not prep.mask[5, 5]
    # roughly the disc the outline encloses
    assert abs(prep.mask.mean() - np.pi * 0.325**2) < 0.03
    # projection texture: opaque paper inside, transparent outside
    assert prep.projection[128, 128].tolist() == [255, 255, 255, 255]
    assert prep.projection[5, 5, 3] == 0


def test_lineart_closes_gaps_in_the_outline(lineart_png):
    broken = lineart_png.copy()
    broken[40:52, 124:132] = 255  # cut a gap in the top of the head outline
    prep = preprocess_lineart(broken)
    assert prep.mask[128, 128]
    assert not prep.mask[5, 5]


def test_lineart_bust_cropped_by_the_canvas_is_still_enclosed():
    img = np.full((200, 200, 4), 255, np.uint8)
    cv2.line(img, (40, 199), (70, 120), (0, 0, 0, 255), 3, cv2.LINE_AA)  # shoulders running off the bottom
    cv2.line(img, (70, 120), (130, 120), (0, 0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(img, (130, 120), (160, 199), (0, 0, 0, 255), 3, cv2.LINE_AA)
    prep = preprocess_lineart(img)
    assert not prep.warnings
    assert prep.mask[170, 100] and not prep.mask[60, 100] and not prep.mask[170, 20]
    assert prep.clean[199, 100, 3] == 0  # the virtual edge is never drawn into the layer


def test_lineart_on_transparent_background(lineart_png):
    ink = 255 - lineart_png[..., 0]
    transparent = np.zeros_like(lineart_png)
    transparent[..., 3] = ink
    a = preprocess_lineart(lineart_png).clean[..., 3].astype(int)
    b = preprocess_lineart(transparent).clean[..., 3].astype(int)
    assert np.abs(a - b).max() <= 1


def test_chroma_matte_solves_edge_alpha_and_removes_spill(cel_png):
    rgb, alpha = chroma_matte(cel_png)
    assert alpha[2, 2] == 0 and alpha[128, 128] == 1
    # Solid interior colours are untouched.
    assert rgb[100, 128].tolist() == list(SKIN)
    assert rgb[162, 150].tolist() == list(BADGE)
    # Edge pixels are partially transparent, not eroded and not hard.
    edge = (alpha > 0.05) & (alpha < 0.95)
    assert edge.sum() > 200
    # No visible pixel is green-dominant any more (the outline is black).
    vis = alpha > 0
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    assert not ((g - np.maximum(r, b))[vis] > 8).any()
    # Compositing the matte back over the key reproduces the source.
    comp = rgb * alpha[..., None] + np.array(KEY) * (1 - alpha[..., None])
    assert np.abs(comp - cel_png[..., :3]).max() < 6


def test_chroma_keeps_the_outline_width(cel_png):
    _, alpha = chroma_matte(cel_png)
    # Coverage summed along a row equals the source figure's width to sub-pixel accuracy.
    src = np.linalg.norm(cel_png[128, :, :3].astype(float) - KEY, axis=-1) / np.linalg.norm(KEY)
    assert abs(alpha[128].sum() - np.clip(src, 0, 1).sum()) < 0.5


def test_cel_projection_has_no_green_outside_the_matte(cel_png):
    prep = preprocess_cel_chroma(cel_png)
    outside = prep.projection[prep.projection[..., 3] == 0][:, :3].astype(int)
    assert len(outside) > 0
    assert not ((outside[:, 1] - np.maximum(outside[:, 0], outside[:, 2])) > 8).any()


def _mask(h=64, w=64):
    m = np.zeros((h, w, 4), np.uint8)
    m[8:56, 8:56, 3] = 255  # coverage
    m[8:56, 40:56, 0] = 255  # disoccluded strip
    m[8:56, 30:40, 1] = 230  # grazing but visible
    return m


def test_denoise_map_protects_visible_detail():
    s = build_denoise_map(_mask(), DenoiseParams.for_mode("cel_chroma"))
    assert s[2, 2] == 0  # outside the object
    assert s[32, 14] <= 0.15  # well-seen surface: badge-safe
    assert 0.3 <= s[32, 33] <= 0.5  # stretched surface
    assert s[32, 50] >= 0.8  # disoccluded
    assert s[32, 14] < s[32, 33] < s[32, 50]


def test_denoise_overrides():
    opts = RefineOptions(denoise_visible_min=0.0, denoise_disoccluded=0.6, feather_px=0)
    s = build_denoise_map(_mask(), DenoiseParams.for_mode("lineart", opts))
    assert s[32, 14] == 0.0
    assert abs(s[32, 50] - 0.6) < 1e-6


def test_postprocess_cel_composites_on_exact_key(cel_png):
    prep = preprocess_cel_chroma(cel_png)
    layer, flat = postprocess_cel_chroma(prep.clean[..., :3], prep.clean[..., 3])
    assert (flat[prep.clean[..., 3] == 0] == KEY).all()
    assert flat[100, 128].tolist() == list(SKIN)
    assert np.abs(flat.astype(int) - cel_png[..., :3]).max() < 8  # round trip
    assert layer.shape[2] == 4


def test_postprocess_lineart_is_crisp_black_on_transparent(lineart_png):
    blurred = cv2.GaussianBlur(lineart_png[..., :3], (0, 0), 1.2)  # stand-in for warp softness
    cover = np.full(blurred.shape[:2], 255, np.uint8)
    layer, white = postprocess_lineart(blurred, cover)
    assert layer[..., :3].max() == 0
    a = layer[..., 3]
    soft_before = ((blurred[..., 0] > 20) & (blurred[..., 0] < 235)).sum()
    soft_after = ((a > 20) & (a < 235)).sum()
    assert soft_after < 0.6 * soft_before  # fewer grey pixels: edges got harder
    assert a.max() == 255 and white[128, 128].tolist() == [255, 255, 255]
