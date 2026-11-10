// CPU resolve of the supersampled viewport capture.
//
// The four render targets are drawn without blending or MSAA, because packed
// depth, normals and mask flags are not values a GPU resolve may average.
// Anti-aliasing comes from rendering at `ss`x and reducing here, with a filter
// that is correct for each kind of data.

export interface RawCapture {
  /** All four are RGBA8, `sw` x `sh`, bottom row first (as read from GL). */
  color: Uint8Array;
  depth: Uint8Array; // R = high byte, G = low byte of 16-bit linear depth
  normal: Uint8Array;
  mask: Uint8Array; // R disoccluded, G stretch, B occluded
}

export interface ResolvedCapture {
  width: number;
  height: number;
  /** Straight alpha; alpha is the fraction of covered subsamples. */
  rgba: Uint8Array;
  /** Linear depth, 0 = near plane, 65535 = far plane or background. */
  depth: Uint16Array;
  normal: Uint8Array;
  mask: Uint8Array;
}

export function resolveCapture(raw: RawCapture, sw: number, sh: number, ss: number): ResolvedCapture {
  const width = Math.floor(sw / ss);
  const height = Math.floor(sh / ss);
  const rgba = new Uint8Array(width * height * 4);
  const depth = new Uint16Array(width * height).fill(65535);
  const normal = new Uint8Array(width * height * 4);
  const mask = new Uint8Array(width * height * 4);
  const total = ss * ss;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let n = 0;
      let r = 0, g = 0, b = 0;
      let nx = 0, ny = 0, nz = 0;
      let m0 = 0, m1 = 0, m2 = 0;
      let nearest = 65535;
      for (let sy = 0; sy < ss; sy++) {
        // GL rows run bottom-up; output rows run top-down.
        const row = sh - 1 - (y * ss + sy);
        for (let sx = 0; sx < ss; sx++) {
          const i = (row * sw + x * ss + sx) * 4;
          if (raw.color[i + 3] === 0) continue;
          n++;
          r += raw.color[i];
          g += raw.color[i + 1];
          b += raw.color[i + 2];
          // Depth: nearest covered sample. Averaging across a silhouette
          // would invent a surface halfway between object and background.
          const d = raw.depth[i] * 256 + raw.depth[i + 1];
          if (d < nearest) nearest = d;
          nx += raw.normal[i] - 127.5;
          ny += raw.normal[i + 1] - 127.5;
          nz += raw.normal[i + 2] - 127.5;
          m0 += raw.mask[i];
          m1 += raw.mask[i + 1];
          m2 += raw.mask[i + 2];
        }
      }
      const o = (y * width + x) * 4;
      if (n === 0) {
        normal[o] = 128;
        normal[o + 1] = 128;
        normal[o + 2] = 255;
        normal[o + 3] = 255;
        continue;
      }
      // Colour and flags average over covered samples only, so edge pixels
      // carry pure object colour and their alpha carries the coverage.
      rgba[o] = Math.round(r / n);
      rgba[o + 1] = Math.round(g / n);
      rgba[o + 2] = Math.round(b / n);
      rgba[o + 3] = Math.round((n / total) * 255);
      depth[y * width + x] = nearest;
      const len = Math.hypot(nx, ny, nz) || 1;
      normal[o] = Math.round((nx / len) * 127.5 + 127.5);
      normal[o + 1] = Math.round((ny / len) * 127.5 + 127.5);
      normal[o + 2] = Math.round((nz / len) * 127.5 + 127.5);
      normal[o + 3] = 255;
      mask[o] = Math.round(m0 / n);
      mask[o + 1] = Math.round(m1 / n);
      mask[o + 2] = Math.round(m2 / n);
      mask[o + 3] = rgba[o + 3];
    }
  }
  return { width, height, rgba, depth, normal, mask };
}
