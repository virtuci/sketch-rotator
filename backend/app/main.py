"""TurnSketch Studio API gateway.

REST + WebSocket surface for the web UI and for DCC bridge plugins. The
gateway does no image work itself: it validates and stores uploads, hands
jobs to the queue, and relays their progress.
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError

from .config import get_settings
from .imaging import io
from .jobs import JobNotFound, make_queue, wait_result
from .schemas import (
    CameraMeta,
    InputMode,
    LayerSpec,
    ProxyMode,
    RefineAccepted,
    RefineOptions,
    SessionInitResponse,
)
from .storage import SessionNotFound, Storage

settings = get_settings()
storage = Storage(settings.data_dir)
REFINE_BUFFERS = ("projected_rgba", "viewport_depth", "viewport_normal", "disocclusion_mask")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.queue = make_queue(settings)
    yield
    await app.state.queue.close()


app = FastAPI(title="TurnSketch Studio", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")


async def _read_image(upload: UploadFile, label: str) -> bytes:
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"{label} is larger than the upload limit")
    return data


def _parse(model, raw: str, label: str):
    try:
        return TypeAdapter(model).validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(422, f"invalid {label}: {exc.errors()[0]['msg']}") from exc


def _job_or_http_error(event: dict) -> dict:
    if event["type"] == "error":
        raise HTTPException(503 if event.get("retryable") else 422, event["message"])
    return event["result"]


@app.get("/api/v1/health")
async def health() -> dict:
    return {
        "status": "ok",
        "queue": settings.queue_backend,
        "refiner": settings.refiner,
        "reconstructor": settings.reconstructor,
        "depth_estimator": settings.depth_estimator,
        "segmenter": settings.segmenter,
    }


@app.post("/api/v1/session/init", response_model=SessionInitResponse)
async def session_init(
    image: Annotated[UploadFile, File()],
    input_mode: Annotated[InputMode, Form()],
    proxy_mode: Annotated[ProxyMode, Form()] = "full_3d_reconstruction",
    separate_props: Annotated[bool, Form()] = False,
    origin_focal_mm: Annotated[float, Form(ge=15.0, le=400.0)] = 85.0,
    layers: Annotated[str, Form()] = "[]",
    layer_masks: Annotated[list[UploadFile], File()] = [],
):
    """Strip the background and build the 3D proxy. Responds once the proxy is ready."""
    specs = _parse(list[LayerSpec], layers, "layers")
    data = await _read_image(image, "image")
    try:
        io.decode_rgba(data, settings.max_image_side)  # reject bad uploads before queueing
    except io.ImageError as exc:
        raise HTTPException(422, str(exc)) from exc

    sid = storage.create()
    sdir = storage.dir(sid)
    (sdir / "input.png").write_bytes(data)
    for i, upload in enumerate(layer_masks):
        (sdir / f"layer_mask_{i}.png").write_bytes(await _read_image(upload, "layer mask"))

    queue = app.state.queue
    job_id = await queue.submit(
        "init_session",
        {
            "session_id": sid,
            "input_mode": input_mode,
            "proxy_mode": proxy_mode,
            "separate_props": separate_props,
            "origin_focal_mm": origin_focal_mm,
            "layers": [s.model_dump() for s in specs],
        },
    )
    return _job_or_http_error(await wait_result(queue, job_id))


@app.get("/api/v1/session/{session_id}", response_model=SessionInitResponse)
async def session_get(session_id: str):
    try:
        return storage.read_meta(session_id)
    except SessionNotFound:
        raise HTTPException(404, "session not found") from None


@app.post("/api/v1/session/{session_id}/refine", response_model=RefineAccepted, status_code=202)
async def session_refine(
    session_id: str,
    projected_rgba: Annotated[UploadFile, File()],
    viewport_depth: Annotated[UploadFile, File()],
    viewport_normal: Annotated[UploadFile, File()],
    disocclusion_mask: Annotated[UploadFile, File()],
    camera: Annotated[str, Form()],
    options: Annotated[str, Form()] = "{}",
):
    """Queue a refinement of the captured viewport buffers. Follow it on the returned WebSocket."""
    try:
        sdir = storage.dir(session_id)
        storage.read_meta(session_id)
    except SessionNotFound:
        raise HTTPException(404, "session not found") from None
    cam = _parse(CameraMeta, camera, "camera")
    opts = _parse(RefineOptions, options, "options")

    job_id = uuid.uuid4().hex
    jdir = sdir / "refine" / job_id
    jdir.mkdir(parents=True)
    uploads = dict(zip(REFINE_BUFFERS, (projected_rgba, viewport_depth, viewport_normal, disocclusion_mask)))
    for name, upload in uploads.items():
        (jdir / f"{name}.png").write_bytes(await _read_image(upload, name))

    await app.state.queue.submit(
        "refine_rotation",
        {"session_id": session_id, "job_id": job_id, "camera": cam.model_dump(), "options": opts.model_dump(exclude_none=True)},
        job_id=job_id,
    )
    return RefineAccepted(
        job_id=job_id, ws_url=f"/api/v1/jobs/{job_id}/ws", status_url=f"/api/v1/jobs/{job_id}"
    )


@app.get("/api/v1/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    """Polling alternative to the WebSocket: the latest event for a job."""
    try:
        return await app.state.queue.latest(job_id)
    except JobNotFound:
        raise HTTPException(404, "job not found") from None


@app.websocket("/api/v1/jobs/{job_id}/ws")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    try:
        async for event in app.state.queue.stream(job_id):
            await ws.send_text(json.dumps(event))
        await ws.close()
    except JobNotFound:
        await ws.close(code=4404, reason="job not found")
    except WebSocketDisconnect:
        pass
