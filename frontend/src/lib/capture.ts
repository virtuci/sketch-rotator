import {
  Color,
  NearestFilter,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { encodePngBlob } from "./png";
import { resolveCapture } from "./resolve";

export interface CaptureBuffers {
  width: number;
  height: number;
  projectedRgba: Blob;
  depth: Blob; // 16-bit greyscale PNG
  normal: Blob;
  mask: Blob;
}

const MAX_TARGET_SIDE = 4096;

/** Largest supersampling factor (at most `wanted`) the GPU and memory budget allow. */
export function pickSupersample(width: number, height: number, maxTextureSize: number, wanted = 2): number {
  const limit = Math.min(maxTextureSize, MAX_TARGET_SIDE);
  let ss = wanted;
  while (ss > 1 && Math.max(width, height) * ss > limit) ss--;
  return ss;
}

/**
 * Render the scene once into four attachments and return them as PNGs.
 *
 * `materials` are the projective materials in the scene; they are switched to
 * export mode for the duration of the render and restored afterwards.
 */
export async function captureBuffers(
  gl: WebGLRenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  materials: ShaderMaterial[],
  width: number,
  height: number,
): Promise<CaptureBuffers> {
  const ss = pickSupersample(width, height, gl.capabilities.maxTextureSize);
  const sw = width * ss;
  const sh = height * ss;

  const target = new WebGLRenderTarget(sw, sh, {
    count: 4,
    depthBuffer: true,
    minFilter: NearestFilter,
    magFilter: NearestFilter,
    generateMipmaps: false,
  });

  const prevTarget = gl.getRenderTarget();
  const prevClear = gl.getClearColor(new Color());
  const prevAlpha = gl.getClearAlpha();
  const raw: Uint8Array[] = [];
  try {
    for (const m of materials) {
      m.uniforms.uExport.value = true;
      m.uniforms.uNear.value = camera.near;
      m.uniforms.uFar.value = camera.far;
    }
    gl.setRenderTarget(target);
    gl.setClearColor(0x000000, 0);
    gl.clear();
    gl.render(scene, camera);

    // three (r172) reads only attachment 0, so select each buffer directly.
    gl.setRenderTarget(target);
    const ctx = gl.getContext() as WebGL2RenderingContext;
    for (let i = 0; i < 4; i++) {
      const pixels = new Uint8Array(sw * sh * 4);
      ctx.readBuffer(ctx.COLOR_ATTACHMENT0 + i);
      ctx.readPixels(0, 0, sw, sh, ctx.RGBA, ctx.UNSIGNED_BYTE, pixels);
      raw.push(pixels);
    }
    ctx.readBuffer(ctx.COLOR_ATTACHMENT0);
  } finally {
    for (const m of materials) m.uniforms.uExport.value = false;
    gl.setRenderTarget(prevTarget);
    gl.setClearColor(prevClear, prevAlpha);
    target.dispose();
  }

  const out = resolveCapture({ color: raw[0], depth: raw[1], normal: raw[2], mask: raw[3] }, sw, sh, ss);
  const [projectedRgba, depth, normal, mask] = await Promise.all([
    encodePngBlob(width, height, { kind: "rgba8", data: out.rgba }),
    encodePngBlob(width, height, { kind: "gray16", data: out.depth }),
    encodePngBlob(width, height, { kind: "rgba8", data: out.normal }),
    encodePngBlob(width, height, { kind: "rgba8", data: out.mask }),
  ]);
  return { width, height, projectedRgba, depth, normal, mask };
}
