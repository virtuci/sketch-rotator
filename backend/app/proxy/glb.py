"""Minimal binary glTF 2.0 writer: one node per mesh, positions + normals + indices."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .mesh import Mesh

_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def write_glb(path: Path, meshes: list[Mesh]) -> None:
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(data: np.ndarray, target: int, component: int, kind: str, bounds: bool = False) -> int:
        raw = np.ascontiguousarray(data).tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw), "target": target})
        blob.extend(raw)
        blob.extend(b"\0" * (-len(blob) % 4))
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": int(data.shape[0]), "type": kind}
        if bounds:
            acc["min"] = data.min(0).astype(float).tolist()
            acc["max"] = data.max(0).astype(float).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    gl_meshes, nodes = [], []
    for mesh in meshes:
        if len(mesh.faces) == 0:
            continue
        pos = add(mesh.vertices.astype("<f4"), _ARRAY_BUFFER, _FLOAT, "VEC3", bounds=True)
        nrm = add(mesh.normals.astype("<f4"), _ARRAY_BUFFER, _FLOAT, "VEC3")
        idx = add(mesh.faces.astype("<u4").reshape(-1), _ELEMENT_ARRAY_BUFFER, _UINT32, "SCALAR")
        gl_meshes.append({"name": mesh.name, "primitives": [{"attributes": {"POSITION": pos, "NORMAL": nrm}, "indices": idx}]})
        nodes.append({"name": mesh.name, "mesh": len(gl_meshes) - 1})

    doc = {
        "asset": {"version": "2.0", "generator": "TurnSketch Studio"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0"))
        f.write(blob)
