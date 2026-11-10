import type { CaptureBuffers } from "./capture";
import type {
  CameraMeta,
  InputMode,
  JobEvent,
  LayerSpec,
  ProxyMode,
  RefineOptions,
  RefineResult,
  Session,
} from "./types";

/**
 * Base URL of the gateway. Empty means same origin, which is how the Docker
 * deployment runs (the reverse proxy routes /api and /files). In local dev it
 * points straight at the FastAPI port.
 */
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

export function absoluteUrl(path: string): string {
  return /^https?:/.test(path) ? path : `${API_BASE}${path}`;
}

function wsUrl(path: string): string {
  const base = API_BASE || window.location.origin;
  return base.replace(/^http/, "ws") + path;
}

async function fail(res: Response): Promise<never> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // not JSON; keep the status line
  }
  throw new Error(detail);
}

export interface InitRequest {
  image: Blob;
  inputMode: InputMode;
  proxyMode: ProxyMode;
  layers?: { spec: Omit<LayerSpec, "mask_index">; mask: Blob }[];
}

export async function initSession(req: InitRequest): Promise<Session> {
  const form = new FormData();
  form.append("image", req.image, "sketch.png");
  form.append("input_mode", req.inputMode);
  form.append("proxy_mode", req.proxyMode);
  const layers = req.layers ?? [];
  form.append("separate_props", String(layers.length > 0));
  form.append("layers", JSON.stringify(layers.map((l, i) => ({ ...l.spec, mask_index: i }))));
  layers.forEach((l, i) => form.append("layer_masks", l.mask, `mask_${i}.png`));
  const res = await fetch(`${API_BASE}/api/v1/session/init`, { method: "POST", body: form });
  if (!res.ok) await fail(res);
  return res.json();
}

/**
 * Upload the capture and follow the job to completion.
 * `onProgress` receives every progress event; the promise settles on the
 * terminal event. Falls back to polling if the WebSocket cannot be opened.
 */
export async function refineRotation(
  sessionId: string,
  buffers: CaptureBuffers,
  camera: CameraMeta,
  options: RefineOptions,
  onProgress: (progress: number, message: string) => void,
): Promise<RefineResult> {
  const form = new FormData();
  form.append("projected_rgba", buffers.projectedRgba, "projected_rgba.png");
  form.append("viewport_depth", buffers.depth, "viewport_depth.png");
  form.append("viewport_normal", buffers.normal, "viewport_normal.png");
  form.append("disocclusion_mask", buffers.mask, "disocclusion_mask.png");
  form.append("camera", JSON.stringify(camera));
  form.append("options", JSON.stringify(options));
  const res = await fetch(`${API_BASE}/api/v1/session/${sessionId}/refine`, { method: "POST", body: form });
  if (!res.ok) await fail(res);
  const accepted: { ws_url: string; status_url: string } = await res.json();

  const handle = (event: JobEvent): RefineResult | undefined => {
    if (event.type === "progress") onProgress(event.progress, event.message);
    if (event.type === "error") throw new Error(event.message);
    return event.type === "result" ? event.result : undefined;
  };

  try {
    return await new Promise<RefineResult>((resolve, reject) => {
      const ws = new WebSocket(wsUrl(accepted.ws_url));
      let settled = false;
      ws.onmessage = (msg) => {
        try {
          const result = handle(JSON.parse(msg.data));
          if (result) {
            settled = true;
            resolve(result);
          }
        } catch (err) {
          settled = true;
          reject(err);
        }
      };
      ws.onerror = () => {
        if (!settled) reject(new SocketUnavailable());
      };
      ws.onclose = () => {
        if (!settled) reject(new SocketUnavailable());
      };
    });
  } catch (err) {
    if (!(err instanceof SocketUnavailable)) throw err;
  }

  // The stream replays from the start server-side, so polling loses nothing.
  for (;;) {
    const poll = await fetch(`${API_BASE}${accepted.status_url}`);
    if (!poll.ok) await fail(poll);
    const result = handle(await poll.json());
    if (result) return result;
    await new Promise((r) => setTimeout(r, 700));
  }
}

class SocketUnavailable extends Error {}
