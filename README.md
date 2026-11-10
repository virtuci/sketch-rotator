# TurnSketch Studio

Rotate a single 2D sketch or cel-shaded keyframe in 3D and get a clean 2D layer
back from the new camera angle. Self-hosted: a Next.js + Three.js web app in
front of a FastAPI gateway and GPU workers.

The artist pastes a layer (`Ctrl+V`), the server strips the background and
builds a 3D proxy, the browser rotates it at interactive rates with the
original drawing projected onto it, and **Apply rotation** sends four viewport
buffers back for refinement. The result goes back to the clipboard in one click.

## What works today, and what is a hook

Everything in the first table runs with no model weights and no GPU, and is
covered by tests or was exercised end to end in a headless browser.

| Piece | State |
| --- | --- |
| Lineart preprocessing (white or transparent background, gap closing, canvas-cropped busts) | Working, tested |
| `#229919` chroma matting (closed-form edge alpha, spill removal, colour bleed) and re-composite on exact key | Working, tested |
| 3D proxy: silhouette inflation, layered `.glb`, glasses plane, separate prop layers, depth-discontinuity cutting | Working, tested |
| Viewport: yaw / pitch / roll, 15–200mm with dolly-zoom, projective texture + disocclusion shader | Working, run in Chromium |
| MRT capture of `projected_rgba`, 16-bit `viewport_depth`, `viewport_normal`, `disocclusion_mask` | Working, run in Chromium |
| Differential-denoise map (visible ≈ 0.05–0.12, stretched up to 0.35, disoccluded 0.85) | Working, tested |
| REST + WebSocket API, inline queue, Celery + Redis queue | Working, tested (Celery path against a real Redis and worker) |
| Clipboard paste in, clipboard copy out, onion skin, side-by-side | Working, run in Chromium |

These are written as integration points and have **not** been run against real
weights. Each is off by default and switched on per worker with a `TS_*`
variable (see `.env.example`):

| Hook | Switch | File |
| --- | --- | --- |
| SDXL + ControlNet depth/normal + IP-Adapter refiner with differential diffusion | `TS_REFINER=diffusion` | `backend/app/models/refiner.py` |
| Hunyuan3D-2 mesh from the clay render | `TS_RECONSTRUCTOR=hunyuan3d` | `backend/app/models/hooks.py` |
| Depth-Anything-V2 depth for the proxy surface | `TS_DEPTH_ESTIMATOR=depth_anything` | `backend/app/models/hooks.py` |
| SAM-2 layer splitting from point / box prompts | `TS_SEGMENTER=sam2` | `backend/app/models/hooks.py` |

Without them the server says so in its responses and the UI shows it:

- **Proxy.** The mesh is the silhouette inflated into a volume (a limb becomes
  a tube as deep as it is wide, a head a dome). It reprojects exactly onto the
  drawing from the original view and gives believable parallax for moderate
  turns, but it knows nothing about anatomy. `canonical_anime_head` currently
  selects a rounder inflation preset, not a landmark-fitted base mesh.
- **Refine.** Visible pixels are the exact geometric warp. Unseen areas are
  filled with blank paper plus a silhouette outline (lineart) or the blurred
  local colour (cel). Nothing is redrawn.
- **Layers.** Glasses and props are split with painted masks uploaded from the
  side panel, since SAM-2 prompting is not wired into the UI yet.

Also not built yet: Docker images (the compose file validates, the images were
not built here), session expiry on disk, authentication, and the DCC bridge
plugins.

## Layout

```
frontend/                         Next.js (App Router) + Three.js via @react-three/fiber
  src/components/
    RotateViewport.tsx            3D viewport, drag-to-rotate, export frame, capture()
    Studio.tsx                    paste / drop, session state, apply + result flow
    BottomBar.tsx                 Rotate / Tilt / Roll / Perspective, mode, view toggles
    SidePanel.tsx                 proxy mode, layers, layer masks, locks
    ResultPanel.tsx               copy to clipboard, downloads
  src/shaders/
    projectiveDisocclusion.ts     ProjectiveDisocclusionMaterial + origin depth pass
  src/lib/
    camera.ts                     focal -> FOV, dolly-zoom distance, rotation order
    capture.ts                    4-attachment render target, readback
    resolve.ts                    supersample resolve, per-buffer filters
    png.ts                        RGBA8 and 16-bit grey PNG encoder
    api.ts                        REST + WebSocket client
    clipboard.ts                  paste, copy, input-mode detection
  tests/                          vitest: camera math, PNG encoder, resolve

backend/
  app/
    main.py                       FastAPI gateway: REST, WebSocket, file serving
    pipeline.py                   the two jobs: init_session, refine_rotation
    jobs.py                       queue interface; inline and Celery backends
    worker.py                     Celery GPU worker
    schemas.py                    API contracts, origin camera convention
    storage.py                    session files on the shared volume
    imaging/
      preprocess.py               lineart ink extraction, chroma matting
      postprocess.py              line cleanup, composite on #229919
      denoise_map.py              differential-diffusion strength map
    proxy/
      mesh.py                     inflation, shell triangulation, clay render
      builder.py                  layers: base, glasses plane, props
      glb.py                      binary glTF writer
    models/
      refiner.py                  classical fallback, diffusion refiner
      hooks.py                    depth, segmentation, reconstruction loaders
  tests/                          pytest: imaging, proxy geometry, API

docker-compose.yml                frontend, gateway, gpu-worker-N, redis, caddy
deploy/Caddyfile                  TLS + routing for the studio domain
```

## Run it locally

No GPU or Redis needed; jobs run in a thread pool inside the gateway.

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:3000 and paste an image.

Tests:

```bash
cd backend && .venv/bin/python -m pytest
```

```bash
cd frontend && npm test && npm run typecheck
```

## Deploy on the studio server

```bash
cp .env.example .env
```

```bash
docker compose up -d --build
```

One `gpu-worker-N` container per card, each a single Celery process that keeps
its models resident; `--profile gpu1` starts the second. The gateway and all
workers share the `sessions` volume, so only paths and parameters go through
Redis. Caddy serves the studio domain over HTTPS, which the browser requires
before it will write to the clipboard.

## API

All paths are under the gateway origin. Bridge plugins use the same endpoints.

### `POST /api/v1/session/init` (multipart)

| Field | Type | Notes |
| --- | --- | --- |
| `image` | file | PNG or WebP |
| `input_mode` | `lineart` \| `cel_chroma` | |
| `proxy_mode` | `canonical_anime_head` \| `full_3d_reconstruction` | default `full_3d_reconstruction` |
| `origin_focal_mm` | number | lens the drawing is assumed to use, default 85 |
| `layers` | JSON | `[{"name","kind":"plane"\|"inflate","offset","locked","mask_index"}]`, or `points` / `box` prompts for SAM-2 |
| `layer_masks` | files | painted masks, same size as the image, indexed by `mask_index` |
| `separate_props` | bool | |

Responds when the proxy is ready with `session_id`, `texture_url` (clean RGBA),
`mesh_url` (`.glb`, one node per layer), `layers[]` (node name, per-layer
projection texture, lock state), `origin_camera`, `proxy_source` and
`warnings[]`.

### `POST /api/v1/session/{session_id}/refine` (multipart) → `202`

Files `projected_rgba`, `viewport_depth` (16-bit grey, 0 = near plane),
`viewport_normal` (view space), `disocclusion_mask` (R disoccluded, G stretch,
B occluded, A coverage); `camera` JSON (angles, focal, fov, distance, near,
far, size); optional `options` JSON (`denoise_visible_min`,
`denoise_visible_max`, `denoise_disoccluded`, `feather_px`, `seed`, `prompt`,
`steps`).

Returns `{job_id, ws_url, status_url}`.

### `WS /api/v1/jobs/{job_id}/ws`

Streams JSON events and closes after the last one. The stream replays from the
start, so connecting late loses nothing.

```
{"type":"queued"}
{"type":"progress","stage":"refine","progress":0.42,"message":"..."}
{"type":"result","result":{"result_rgba_url","result_flat_url","denoise_map_url","refiner","width","height"}}
{"type":"error","message":"...","retryable":false}
```

`GET /api/v1/jobs/{job_id}` returns the latest event for clients that cannot
hold a socket. `GET /api/v1/health` reports which models are enabled.

## How the pieces fit

**Shared world.** The image plane is at z = 0, 2 units tall, +z toward the
viewer. The origin camera (the viewpoint the sketch was drawn from) sits on +z
at the distance where that plane fills its frame. The backend unprojects every
proxy vertex along its own origin-camera ray, so the mesh lands exactly on the
drawing when seen from there.

**Dolly zoom.** `distance = 1 / tan(fovY / 2)` with `fovY = 2·atan(12 / focal)`.
Changing the lens moves the camera so the image plane keeps filling the frame;
only perspective convergence changes. At the origin focal length and zero
rotation the view camera equals the origin camera and the preview is the
original drawing, pixel for pixel.

**Shader.** Each fragment is transformed back into the un-rotated frame and
projected through the origin camera to find its texel. It is marked disoccluded
if it faces away from the origin camera (`N·V ≤ 0.2`), if a shadow-map lookup
from the origin camera finds something in front of it, or if it is a back face.
Disoccluded fragments do not show the drawing projected through the object;
they get a neutral base for the refiner.

**Capture.** One render into four attachments at 2× resolution, without
blending or MSAA because packed depth and mask flags must not be averaged by
the GPU. The CPU resolve averages colour over covered samples, takes the
nearest depth, renormalises normals, and writes coverage into alpha.

**Protecting detail.** The denoise map keeps well-seen surface at 0.05 (cel) or
0.12 (lineart), so emblems and hard cel boundaries are moved by geometry and
not redrawn. Strength rises only where the surface was seen at a grazing angle
and reaches 0.85 where it was never seen.

**Layers.** A `plane` layer (glasses) is a thin shell floating in front of the
surface it sits on; the face continues underneath and is flagged unseen where
the frames used to cover it, which is what produces frame-to-pupil parallax. An
`inflate` layer (desk, chair, held prop) is cut out of the base and given its
own volume, and can be locked so it stays put while the character turns.
