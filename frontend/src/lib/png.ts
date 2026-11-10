// Minimal PNG encoder.
//
// The canvas API cannot be used for the viewport export: it only writes 8-bit
// images, and it premultiplies alpha, which corrupts colour in low-alpha edge
// pixels. This writes straight RGBA8 and true 16-bit greyscale, compressing
// with the browser's native zlib (CompressionStream).

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array, seed = 0xffffffff): number {
  let c = seed;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return c;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, (crc32(out.subarray(4, 8 + data.length)) ^ 0xffffffff) >>> 0);
  return out;
}

async function zlib(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export type PngPixels =
  | { kind: "rgba8"; data: Uint8Array }
  | { kind: "gray16"; data: Uint16Array };

/** Encode top-to-bottom pixel rows as PNG bytes. */
export async function encodePng(width: number, height: number, pixels: PngPixels): Promise<Uint8Array> {
  const bytesPerPixel = pixels.kind === "rgba8" ? 4 : 2;
  if (pixels.data.length !== width * height * (pixels.kind === "rgba8" ? 4 : 1)) {
    throw new Error("pixel buffer does not match the image size");
  }
  const stride = width * bytesPerPixel;
  const raw = new Uint8Array((stride + 1) * height); // each row: filter byte 0 + pixels
  if (pixels.kind === "rgba8") {
    for (let y = 0; y < height; y++) {
      raw.set(pixels.data.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
    }
  } else {
    const view = new DataView(raw.buffer);
    for (let y = 0; y < height; y++) {
      let o = y * (stride + 1) + 1;
      for (let x = 0; x < width; x++, o += 2) view.setUint16(o, pixels.data[y * width + x]); // big-endian
    }
  }

  const ihdr = new Uint8Array(13);
  const iv = new DataView(ihdr.buffer);
  iv.setUint32(0, width);
  iv.setUint32(4, height);
  ihdr[8] = pixels.kind === "rgba8" ? 8 : 16; // bit depth
  ihdr[9] = pixels.kind === "rgba8" ? 6 : 0; // colour type: RGBA / greyscale

  const parts = [
    new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", await zlib(raw)),
    chunk("IEND", new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

export async function encodePngBlob(width: number, height: number, pixels: PngPixels): Promise<Blob> {
  return new Blob([(await encodePng(width, height, pixels)) as BlobPart], { type: "image/png" });
}
