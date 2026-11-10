import { inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { encodePng } from "@/lib/png";
import { resolveCapture } from "@/lib/resolve";
import { pickSupersample } from "@/lib/capture";

function parse(png: Uint8Array) {
  const view = new DataView(png.buffer, png.byteOffset);
  const chunks: Record<string, Uint8Array> = {};
  for (let o = 8; o < png.length; ) {
    const len = view.getUint32(o);
    const type = String.fromCharCode(...png.subarray(o + 4, o + 8));
    chunks[type] = png.subarray(o + 8, o + 8 + len);
    o += 12 + len;
  }
  return chunks;
}

describe("png encoder", () => {
  it("writes straight RGBA8 without touching low-alpha colour", async () => {
    const data = new Uint8Array([255, 0, 0, 1, 10, 20, 30, 128]);
    const chunks = parse(await encodePng(2, 1, { kind: "rgba8", data }));
    expect([...chunks.IHDR.subarray(8, 10)]).toEqual([8, 6]);
    expect([...inflateSync(chunks.IDAT)]).toEqual([0, ...data]);
  });

  it("writes 16-bit greyscale big-endian", async () => {
    const chunks = parse(await encodePng(2, 2, { kind: "gray16", data: new Uint16Array([0, 258, 40000, 65535]) }));
    expect([...chunks.IHDR.subarray(8, 10)]).toEqual([16, 0]);
    expect([...inflateSync(chunks.IDAT)]).toEqual([0, 0, 0, 1, 2, 0, 0x9c, 0x40, 0xff, 0xff]);
  });

  it("rejects a buffer of the wrong size", async () => {
    await expect(encodePng(2, 2, { kind: "rgba8", data: new Uint8Array(4) })).rejects.toThrow();
  });
});

describe("capture resolve", () => {
  // 2x2 supersamples -> one output pixel. GL order: bottom row first.
  const px = (...v: number[][]) => new Uint8Array(v.flat());
  const empty = [0, 0, 0, 0];

  it("averages colour over covered samples and puts coverage in alpha", () => {
    const out = resolveCapture(
      {
        color: px([200, 100, 0, 255], empty, [100, 100, 0, 255], empty),
        depth: px([0x40, 0x00, 0, 255], empty, [0x20, 0x10, 0, 255], empty),
        normal: px([128, 128, 255, 255], empty, [128, 128, 255, 255], empty),
        mask: px([255, 0, 0, 255], empty, [0, 100, 0, 255], empty),
      },
      2, 2, 2,
    );
    expect([...out.rgba]).toEqual([150, 100, 0, 128]);
    expect(out.depth[0]).toBe(0x2010); // nearest covered sample, not the mean
    expect([...out.normal]).toEqual([128, 128, 255, 255]);
    expect([...out.mask]).toEqual([128, 50, 0, 128]);
  });

  it("leaves uncovered pixels transparent at far depth and flips rows", () => {
    const top = [9, 9, 9, 255];
    const out = resolveCapture(
      { color: px(empty, top), depth: px(empty, [0, 5, 0, 255]), normal: px(empty, top), mask: px(empty, empty) },
      1, 2, 1,
    );
    expect(out.rgba[3]).toBe(255); // GL's last row is the image's first
    expect(out.depth[0]).toBe(5);
    expect(out.rgba[7]).toBe(0);
    expect(out.depth[1]).toBe(65535);
  });

  it("limits supersampling to what the GPU can allocate", () => {
    expect(pickSupersample(1024, 1024, 16384)).toBe(2);
    expect(pickSupersample(3000, 2000, 16384)).toBe(1);
    expect(pickSupersample(1024, 512, 1024)).toBe(1);
  });
});
