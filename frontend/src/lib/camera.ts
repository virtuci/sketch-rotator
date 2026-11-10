import { Euler, MathUtils, PerspectiveCamera } from "three";

/** Full-frame sensor height; focal lengths in the UI are 35mm-equivalent. */
export const SENSOR_HEIGHT_MM = 24;
export const FOCAL_MIN_MM = 15;
export const FOCAL_MAX_MM = 200;

export interface ViewParams {
  /** Degrees. Spin about the object's vertical axis. */
  yaw: number;
  /** Degrees. Tilt toward / away from the camera. */
  pitch: number;
  /** Degrees. Rotation in the picture plane. */
  roll: number;
  /** 35mm-equivalent focal length in millimetres. */
  focal: number;
}

/** Vertical field of view in radians. */
export function focalToFovY(focalMm: number): number {
  return 2 * Math.atan(SENSOR_HEIGHT_MM / (2 * focalMm));
}

/**
 * Dolly-zoom compensation.
 *
 * The image plane (z = 0) is `2 * halfHeight` world units tall. For it to keep
 * filling the frame at any field of view, the camera must sit where the
 * frustum is exactly that tall:
 *
 *     distance = halfHeight / tan(fovY / 2)
 *
 * Shortening the lens widens the frustum, so the camera moves in by the same
 * ratio; the subject keeps its size while perspective convergence changes.
 * At 200mm the camera is 16.7 units away and the view is near-orthographic;
 * at 15mm it is 1.25 units away and anything in front of the plane balloons.
 */
export function dollyDistance(fovY: number, halfHeight = 1): number {
  return halfHeight / Math.tan(fovY / 2);
}

/**
 * Object rotation for the given angles. Order Z·X·Y: yaw spins the object
 * about its own up axis, pitch then tilts it about the screen's horizontal
 * axis, and roll turns the result in the picture plane, so each slider keeps
 * meaning the same thing whatever the others are set to.
 */
export function objectEuler(p: Pick<ViewParams, "yaw" | "pitch" | "roll">): Euler {
  return new Euler(
    MathUtils.degToRad(p.pitch),
    MathUtils.degToRad(p.yaw),
    MathUtils.degToRad(p.roll),
    "ZXY",
  );
}

export interface CameraFit {
  /** Aspect of the surface being rendered to. */
  viewAspect: number;
  /** Aspect of the sketch; the export frame. */
  imageAspect: number;
  /** >1 leaves a border around the export frame (interactive preview only). */
  margin?: number;
  /** Radius of the proxy's bounding sphere about the origin, for near/far. */
  boundsRadius: number;
}

export interface CameraState {
  fovY: number;
  distance: number;
  near: number;
  far: number;
}

/**
 * Place `camera` on +Z for the given focal length with dolly-zoom distance.
 *
 * The dolly distance always comes from the true lens field of view. When the
 * surface is narrower than the sketch, or a preview margin is requested, only
 * the frustum is opened wider; the viewpoint, and therefore the perspective,
 * stays exactly what the export will have.
 */
export function placeCamera(camera: PerspectiveCamera, focalMm: number, fit: CameraFit): CameraState {
  const fovY = focalToFovY(focalMm);
  const distance = dollyDistance(fovY);
  const widen = (fit.margin ?? 1) * Math.max(1, fit.imageAspect / fit.viewAspect);
  // Tight clip planes keep the 16-bit depth export precise.
  const near = Math.max(distance - fit.boundsRadius, 0.05);
  const far = distance + fit.boundsRadius;

  camera.fov = MathUtils.radToDeg(2 * Math.atan(Math.tan(fovY / 2) * widen));
  camera.aspect = fit.viewAspect;
  camera.near = near;
  camera.far = far;
  camera.position.set(0, 0, distance);
  camera.quaternion.identity();
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return { fovY, distance, near, far };
}

/** Slider position in [0, 1] <-> focal length, logarithmic so each stop feels equal. */
export function focalFromSlider(t: number): number {
  return FOCAL_MIN_MM * Math.pow(FOCAL_MAX_MM / FOCAL_MIN_MM, MathUtils.clamp(t, 0, 1));
}

export function sliderFromFocal(focalMm: number): number {
  return Math.log(focalMm / FOCAL_MIN_MM) / Math.log(FOCAL_MAX_MM / FOCAL_MIN_MM);
}
