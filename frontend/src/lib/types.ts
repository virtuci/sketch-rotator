// Mirrors backend/app/schemas.py.

export type InputMode = "lineart" | "cel_chroma";
export type ProxyMode = "canonical_anime_head" | "full_3d_reconstruction";

export interface OriginCamera {
  focal_mm: number;
  fov_y: number;
  aspect: number;
  distance: number;
  near: number;
  far: number;
}

export interface LayerInfo {
  name: string;
  node: string;
  texture_url: string;
  locked: boolean;
  kind: "base" | "plane" | "inflate";
}

export interface Session {
  session_id: string;
  input_mode: InputMode;
  proxy_mode: ProxyMode;
  width: number;
  height: number;
  texture_url: string;
  mesh_url: string;
  layers: LayerInfo[];
  origin_camera: OriginCamera;
  proxy_source: string;
  warnings: string[];
}

export interface LayerSpec {
  name: string;
  kind: "plane" | "inflate";
  offset?: number;
  locked?: boolean;
  mask_index?: number;
}

export interface CameraMeta {
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
  focal_mm: number;
  fov_y: number;
  distance: number;
  near: number;
  far: number;
  width: number;
  height: number;
}

export interface RefineOptions {
  denoise_visible_min?: number;
  denoise_visible_max?: number;
  denoise_disoccluded?: number;
  feather_px?: number;
  seed?: number;
  prompt?: string;
}

export interface RefineResult {
  result_rgba_url: string;
  result_flat_url: string;
  denoise_map_url: string;
  refiner: string;
  width: number;
  height: number;
}

export type JobEvent =
  | { type: "queued" }
  | { type: "progress"; stage: string; progress: number; message: string }
  | { type: "result"; result: RefineResult }
  | { type: "error"; message: string; retryable?: boolean };
