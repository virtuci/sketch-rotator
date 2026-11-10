"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { copyPngToClipboard } from "@/lib/clipboard";
import type { InputMode, RefineResult } from "@/lib/types";

export interface LoadedResult {
  meta: RefineResult;
  layer: Blob;
  flat: Blob;
  layerUrl: string;
  flatUrl: string;
}

const REFINER_NOTE: Record<string, string> = {
  classical:
    "No diffusion model is enabled on the server. Visible pixels are the exact geometric warp; unseen areas are filled from their surroundings, not redrawn.",
  diffusion: "Refined with the diffusion pass.",
};

export function ResultPanel({ result, inputMode, onClose }: { result: LoadedResult; inputMode: InputMode; onClose: () => void }) {
  const [note, setNote] = useState<string | null>(null);
  const cel = inputMode === "cel_chroma";

  const copy = async (blob: Blob, label: string) => {
    try {
      await copyPngToClipboard(blob);
      setNote(`${label} copied. Paste it into your canvas.`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "Copy failed.");
    }
  };

  return (
    <motion.aside
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      data-testid="result-panel"
      className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-white/5 bg-ink-900 p-4 text-sm"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-[11px] uppercase tracking-wide text-white/45">Rotated layer</h2>
        <button type="button" onClick={onClose} aria-label="Close result" className="text-white/45 hover:text-white">×</button>
      </div>
      <div className={`overflow-hidden rounded-lg ${cel ? "checker" : "bg-white"}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={result.layerUrl} alt="Rotated sketch layer" data-testid="result-image" className="block w-full" />
      </div>
      <button type="button" data-testid="copy-layer" onClick={() => copy(result.layer, "Layer")}
        className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white">
        Copy rotated layer to clipboard
      </button>
      {cel && (
        <button type="button" data-testid="copy-flat" onClick={() => copy(result.flat, "Green-screen frame")}
          className="rounded-lg border border-white/10 px-3 py-2 text-xs hover:bg-white/5">
          <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-chroma align-middle" />
          Copy on #229919
        </button>
      )}
      <div className="flex gap-2 text-xs">
        <a href={result.layerUrl} download="turnsketch_layer.png" className="flex-1 rounded-md border border-white/10 px-2 py-1.5 text-center hover:bg-white/5">
          Download layer
        </a>
        <a href={result.flatUrl} download={cel ? "turnsketch_229919.png" : "turnsketch_white.png"}
          className="flex-1 rounded-md border border-white/10 px-2 py-1.5 text-center hover:bg-white/5">
          Download {cel ? "on green" : "on white"}
        </a>
      </div>
      {note && <p role="status" className="text-xs text-white/70">{note}</p>}
      <p className="text-xs text-white/40">
        {result.meta.width}×{result.meta.height} · {REFINER_NOTE[result.meta.refiner] ?? result.meta.refiner}
      </p>
    </motion.aside>
  );
}
