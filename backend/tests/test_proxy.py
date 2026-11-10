import json
import struct

import numpy as np

from app.proxy.builder import build_layers
from app.proxy.glb import write_glb
from app.proxy.mesh import build_shell, inflate_height, register_to_silhouette
from app.schemas import LayerSpec, OriginCamera


def disc(n=128, r=0.3):
    y, x = np.mgrid[0:n, 0:n]
    return (x - n / 2) ** 2 + (y - n / 2) ** 2 < (r * n) ** 2


def project(cam: OriginCamera, v: np.ndarray) -> np.ndarray:
    """Origin-camera NDC of world points."""
    depth = cam.distance - v[:, 2]
    f = 1.0 / np.tan(cam.fov_y / 2)
    return np.stack([v[:, 0] * f / cam.aspect / depth, v[:, 1] * f / depth], 1)


def test_inflate_is_a_dome():
    h = inflate_height(disc(), None)
    assert abs(h.max() - 0.6) < 0.02  # radius 0.3 of height 2 world units
    assert h[64, 64] > h[64, 40] > h[64, 30] > 0
    assert h[0, 0] == 0


def test_shell_reprojects_onto_the_silhouette():
    for focal in (24.0, 85.0):
        mask = disc()
        cam = OriginCamera.for_image(128, 128, focal)
        h = inflate_height(mask, None)
        mesh = build_shell("m", mask, h, -h, cam, max_cells=64)
        ndc = project(cam, mesh.vertices)
        # Every vertex, front or back, lands on the disc (plus the one-cell growth).
        assert np.linalg.norm(ndc, axis=1).max() < 0.6 + 2 * (2 / 64) * 1.5
        # Front faces point at the camera, back faces away.
        front = mesh.vertices[:, 2] > 0.2
        assert (mesh.normals[front, 2] > 0).all()
        assert (mesh.normals[mesh.vertices[:, 2] < -0.2, 2] < 0).all()


def test_discontinuity_cut_prevents_depth_bridging():
    mask = np.ones((64, 64), bool)
    z = np.zeros((64, 64), np.float32)
    z[:, 32:] = 0.5  # a "hand" 0.5 in front of the "torso"
    cam = OriginCamera.for_image(64, 64, 85.0)
    bridged = build_shell("m", mask, z, None, cam, max_cells=64)
    cut = build_shell("m", mask, z, None, cam, max_cells=64, discontinuity=0.12)

    def spans(mesh):
        zs = mesh.vertices[mesh.faces][:, :, 2]
        return zs.max(1) - zs.min(1)

    assert spans(bridged).max() > 0.4  # without the cut, wall triangles join the two
    assert spans(cut).max() < 0.12
    assert len(cut.faces) < len(bridged.faces)


def test_glasses_plane_floats_in_front_of_the_face():
    mask = disc()
    glasses = np.zeros_like(mask)
    glasses[56:64, 40:88] = True
    cam = OriginCamera.for_image(128, 128, 85.0)
    layers = build_layers(mask, cam, "canonical_anime_head", [(LayerSpec(name="glasses", offset=0.06), glasses)])
    base, g = layers
    assert g.kind == "plane" and base.mask.sum() == mask.sum()  # face continues under the frames
    face_z = base.mesh.vertices[:, 2].max()
    gz = g.mesh.vertices[:, 2]
    assert gz.max() - gz.min() < 0.02  # flat
    assert gz.min() > face_z * 0.9  # in front of the surface it sits on


def test_inflate_layer_is_removed_from_the_base():
    mask = disc()
    desk = np.zeros_like(mask)
    desk[90:128, :] = True
    desk &= mask
    cam = OriginCamera.for_image(128, 128, 85.0)
    spec = LayerSpec(name="desk", kind="inflate", locked=True, offset=0.0)
    base, d = build_layers(mask, cam, "full_3d_reconstruction", [(spec, desk)])
    assert not (base.mask & desk).any() and d.locked


def test_glb_container(tmp_path):
    mask = disc()
    cam = OriginCamera.for_image(128, 128, 85.0)
    h = inflate_height(mask, None)
    mesh = build_shell("layer_base", mask, h, -h, cam, max_cells=32)
    path = tmp_path / "p.glb"
    write_glb(path, [mesh])
    raw = path.read_bytes()
    magic, version, total = struct.unpack("<4sII", raw[:12])
    assert magic == b"glTF" and version == 2 and total == len(raw)
    jlen, jtype = struct.unpack("<I4s", raw[12:20])
    doc = json.loads(raw[20 : 20 + jlen])
    assert jtype == b"JSON" and doc["nodes"][0]["name"] == "layer_base"
    blen, btype = struct.unpack("<I4s", raw[20 + jlen : 28 + jlen])
    assert btype == b"BIN\0" and doc["buffers"][0]["byteLength"] == blen
    idx = doc["accessors"][2]
    view = doc["bufferViews"][idx["bufferView"]]
    start = 28 + jlen + view["byteOffset"]
    indices = np.frombuffer(raw[start : start + view["byteLength"]], "<u4")
    assert len(indices) == mesh.faces.size and indices.max() == len(mesh.vertices) - 1


def test_register_external_mesh_to_silhouette():
    mask = np.zeros((100, 200), bool)
    mask[20:80, 120:160] = True  # tall box on the right
    cam = OriginCamera.for_image(200, 100, 85.0)
    rng = np.random.default_rng(0)
    verts = rng.uniform(-1, 1, (500, 3)).astype(np.float32) * [0.5, 0.75, 0.4]
    out = register_to_silhouette(verts, mask, cam)
    assert abs(out[:, 1].max() - 0.6) < 0.01 and abs(out[:, 1].min() + 0.6) < 0.01
    assert abs((out[:, 0].max() + out[:, 0].min()) / 2 - 0.8) < 0.01  # centred on the box (x in [-2, 2])
