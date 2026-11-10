import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TS_DATA_DIR", tempfile.mkdtemp(prefix="turnsketch-test-"))
os.environ.setdefault("TS_QUEUE_BACKEND", "inline")

KEY = (0x22, 0x99, 0x19)
SKIN = (250, 214, 190)
SHIRT = (60, 80, 160)
BADGE = (230, 40, 40)


def _aa(draw, size, scale=4):
    """Render at `scale`x and box-filter down, giving true anti-aliased edges."""
    big = draw(size * scale, scale)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="session")
def lineart_png() -> np.ndarray:
    """A head outline with two eyes: black ~3px lines on white, RGBA."""
    def draw(n, s):
        img = np.full((n, n, 3), 255, np.uint8)
        cv2.circle(img, (n // 2, n // 2), int(n * 0.32), (0, 0, 0), 3 * s, cv2.LINE_AA)
        for dx in (-0.12, 0.12):
            cv2.circle(img, (int(n * (0.5 + dx)), int(n * 0.45)), int(n * 0.04), (0, 0, 0), 3 * s, cv2.LINE_AA)
        return img
    rgb = _aa(draw, 256)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


@pytest.fixture(scope="session")
def cel_png() -> np.ndarray:
    """A two-tone figure with a black outline and a small badge on #229919, RGBA."""
    def draw(n, s):
        img = np.full((n, n, 3), KEY, np.uint8)
        c = (n // 2, n // 2)
        cv2.circle(img, c, int(n * 0.34), (0, 0, 0), -1)
        cv2.circle(img, c, int(n * 0.34) - 3 * s, SKIN, -1)
        cv2.rectangle(img, (int(n * 0.3), int(n * 0.55)), (int(n * 0.7), int(n * 0.75)), SHIRT, -1)
        cv2.rectangle(img, (int(n * 0.55), int(n * 0.6)), (int(n * 0.62), int(n * 0.67)), BADGE, -1)
        return img
    rgb = _aa(draw, 256)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def encode_png(rgba: np.ndarray) -> bytes:
    code = cv2.COLOR_RGBA2BGRA if rgba.ndim == 3 and rgba.shape[2] == 4 else None
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, code) if code else rgba)
    assert ok
    return buf.tobytes()
