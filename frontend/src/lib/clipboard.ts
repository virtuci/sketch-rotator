import type { InputMode } from "./types";

/** First image in a paste or drop payload, if any. */
export function imageFromTransfer(data: DataTransfer | null): File | null {
  if (!data) return null;
  for (const file of Array.from(data.files)) {
    if (file.type.startsWith("image/")) return file;
  }
  for (const item of Array.from(data.items)) {
    if (item.kind === "file" && item.type.startsWith("image/")) return item.getAsFile();
  }
  return null;
}

/**
 * Write a PNG to the system clipboard. Browsers only allow this in a secure
 * context (HTTPS or localhost) and from a user gesture, so the studio domain
 * must be served over TLS for the one-click copy to work.
 */
export async function copyPngToClipboard(blob: Blob): Promise<void> {
  if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
    throw new Error("Clipboard write is unavailable. Open the studio over HTTPS, or download the PNG instead.");
  }
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

const KEY = [0x22, 0x99, 0x19];

/** Guess the input mode from the image border: mostly #229919 means a chroma keyframe. */
export async function detectInputMode(image: Blob): Promise<InputMode> {
  const bitmap = await createImageBitmap(image);
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return "lineart";
  ctx.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();
  const { data } = ctx.getImageData(0, 0, size, size);
  let border = 0;
  let green = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x > 1 && x < size - 2 && y > 1 && y < size - 2) continue;
      const i = (y * size + x) * 4;
      border++;
      const d = Math.hypot(data[i] - KEY[0], data[i + 1] - KEY[1], data[i + 2] - KEY[2]);
      if (data[i + 3] > 200 && d < 40) green++;
    }
  }
  return green / border > 0.5 ? "cel_chroma" : "lineart";
}
