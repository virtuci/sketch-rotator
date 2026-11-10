import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from .conftest import KEY, encode_png


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _init(client, png, mode, **extra):
    return client.post(
        "/api/v1/session/init",
        files={"image": ("sketch.png", encode_png(png), "image/png"), **extra.pop("files", {})},
        data={"input_mode": mode, **extra},
    )


def _buffers(n=256):
    """Synthetic capture: a disc whose right third is disoccluded."""
    y, x = np.mgrid[0:n, 0:n]
    inside = (x - n / 2) ** 2 + (y - n / 2) ** 2 < (0.3 * n) ** 2
    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[inside] = (240, 200, 180, 255)
    rgba[100:110, 90:100, :3] = (230, 40, 40)  # badge
    mask = np.zeros((n, n, 4), np.uint8)
    mask[inside, 3] = 255
    mask[inside & (x > n * 0.62), 0] = 255
    normal = np.zeros((n, n, 4), np.uint8)
    normal[inside] = (128, 128, 255, 255)
    depth = np.full((n, n), 65535, np.uint16)
    depth[inside] = 20000
    ok, depth_png = cv2.imencode(".png", depth)
    assert ok
    return {
        "projected_rgba": ("a.png", encode_png(rgba), "image/png"),
        "viewport_depth": ("b.png", depth_png.tobytes(), "image/png"),
        "viewport_normal": ("c.png", encode_png(normal), "image/png"),
        "disocclusion_mask": ("d.png", encode_png(mask), "image/png"),
    }


CAMERA = dict(yaw_deg=30, pitch_deg=0, roll_deg=0, focal_mm=50, fov_y=0.47, distance=4.1, near=1.6, far=6.6, width=256, height=256)


def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


@pytest.mark.parametrize("mode", ["lineart", "cel_chroma"])
def test_init_returns_texture_and_mesh(client, lineart_png, cel_png, mode):
    r = _init(client, lineart_png if mode == "lineart" else cel_png, mode)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 256 and body["layers"][0]["node"] == "layer_base"
    assert client.get(body["mesh_url"]).content[:4] == b"glTF"
    assert client.get(body["texture_url"]).status_code == 200
    assert client.get(body["layers"][0]["texture_url"]).status_code == 200
    assert client.get(f"/api/v1/session/{body['session_id']}").json()["session_id"] == body["session_id"]


def test_init_with_glasses_layer_mask(client, cel_png):
    m = np.zeros((256, 256, 4), np.uint8)
    m[100:112, 80:176] = 255
    spec = [{"name": "glasses", "kind": "plane", "mask_index": 0}]
    r = _init(client, cel_png, "cel_chroma", layers=json.dumps(spec),
              files={"layer_masks": ("g.png", encode_png(m), "image/png")})
    assert r.status_code == 200, r.text
    assert [layer["name"] for layer in r.json()["layers"]] == ["base", "glasses"]


def test_init_rejects_bad_input(client, cel_png):
    r = client.post("/api/v1/session/init", files={"image": ("x.png", b"not an image", "image/png")},
                    data={"input_mode": "lineart"})
    assert r.status_code == 422
    assert _init(client, cel_png, "watercolor").status_code == 422
    blank = np.full((64, 64, 4), 255, np.uint8)
    blank[..., :3] = KEY
    r = _init(client, blank, "cel_chroma")
    assert r.status_code == 422 and "background removal" in r.json()["detail"]


def test_refine_streams_progress_and_protects_visible_pixels(client, cel_png):
    sid = _init(client, cel_png, "cel_chroma").json()["session_id"]
    r = client.post(f"/api/v1/session/{sid}/refine", files=_buffers(), data={"camera": json.dumps(CAMERA)})
    assert r.status_code == 202, r.text
    accepted = r.json()
    events = []
    with client.websocket_connect(accepted["ws_url"]) as ws:
        while True:
            events.append(json.loads(ws.receive_text()))
            if events[-1]["type"] in ("result", "error"):
                break
    assert events[0]["type"] == "queued"
    progress = [e["progress"] for e in events if e["type"] == "progress"]
    assert progress == sorted(progress) and progress[-1] == 1.0
    result = events[-1]["result"]
    assert client.get(accepted["status_url"]).json()["type"] == "result"

    def fetch(url, flags=cv2.IMREAD_UNCHANGED):
        return cv2.imdecode(np.frombuffer(client.get(url).content, np.uint8), flags)

    flat = fetch(result["result_flat_url"])[..., ::-1]
    assert flat[2, 2].tolist() == list(KEY)  # exact key outside
    assert flat[105, 95].tolist() == [230, 40, 40]  # visible badge is pixel-identical
    assert flat[128, 190].tolist() != list(KEY)  # disoccluded area was filled, not left empty
    layer = fetch(result["result_rgba_url"])
    assert layer.shape == (256, 256, 4) and layer[2, 2, 3] == 0
    dmap = fetch(result["denoise_map_url"])
    assert dmap[105, 95] <= 0.15 * 255 and dmap[128, 190] >= 0.8 * 255


def test_refine_rejects_mismatched_buffers_and_unknown_ids(client, lineart_png):
    sid = _init(client, lineart_png, "lineart").json()["session_id"]
    bad = dict(CAMERA, width=512)
    r = client.post(f"/api/v1/session/{sid}/refine", files=_buffers(), data={"camera": json.dumps(bad)})
    with client.websocket_connect(r.json()["ws_url"]) as ws:
        while (event := json.loads(ws.receive_text()))["type"] not in ("result", "error"):
            pass
    assert event["type"] == "error" and "256x256" in event["message"]
    assert client.post("/api/v1/session/" + "0" * 32 + "/refine", files=_buffers(),
                       data={"camera": json.dumps(CAMERA)}).status_code == 404
    assert client.post("/api/v1/session/..%2F..%2Fetc/refine", files=_buffers(),
                       data={"camera": json.dumps(CAMERA)}).status_code == 404
    assert client.get("/api/v1/jobs/nope").status_code == 404
