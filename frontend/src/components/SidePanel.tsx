"use client";

import { useRef } from "react";
import type { ProxyMode, Session } from "@/lib/types";

export interface ExtraLayer {
  id: string;
  name: string;
  kind: "plane" | "inflate";
  locked: boolean;
  mask: File;
}

interface Props {
  session: Session | null;
  proxyMode: ProxyMode;
  onProxyMode: (m: ProxyMode) => void;
  extraLayers: ExtraLayer[];
  onExtraLayers: (layers: ExtraLayer[]) => void;
  lockedLayers: Record<string, boolean>;
  onLockedLayers: (l: Record<string, boolean>) => void;
  dirty: boolean;
  busy: boolean;
  onRebuild: () => void;
}

const PROXY_LABEL: Record<ProxyMode, string> = {
  canonical_anime_head: "Head / bust",
  full_3d_reconstruction: "Full figure",
};

export function SidePanel(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const pendingKind = useRef<"plane" | "inflate">("plane");

  const addMask = (kind: "plane" | "inflate") => {
    pendingKind.current = kind;
    fileRef.current?.click();
  };
  const onFile = (file: File | undefined) => {
    if (!file) return;
    const kind = pendingKind.current;
    const base = kind === "plane" ? "glasses" : "prop";
    const taken = new Set(p.extraLayers.map((l) => l.name));
    let name = base;
    for (let i = 2; taken.has(name); i++) name = `${base}_${i}`;
    p.onExtraLayers([...p.extraLayers, { id: crypto.randomUUID(), name, kind, locked: kind === "inflate", mask: file }]);
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col gap-5 overflow-y-auto border-r border-white/5 bg-ink-900 p-4 text-sm">
      <section>
        <h2 className="mb-2 text-[11px] uppercase tracking-wide text-white/45">3D proxy</h2>
        <div className="flex gap-1 rounded-lg bg-black/25 p-0.5">
          {(Object.keys(PROXY_LABEL) as ProxyMode[]).map((m) => (
            <button
              key={m}
              type="button"
              data-testid={`proxy-${m}`}
              aria-pressed={p.proxyMode === m}
              onClick={() => p.onProxyMode(m)}
              className={`flex-1 rounded-md px-2 py-1.5 text-xs ${
                p.proxyMode === m ? "bg-accent/25 text-accent-soft" : "text-white/60 hover:text-white"
              }`}
            >
              {PROXY_LABEL[m]}
            </button>
          ))}
        </div>
        {p.session && <p className="mt-2 text-xs text-white/40">Built with: {p.session.proxy_source}</p>}
      </section>

      <section>
        <h2 className="mb-2 text-[11px] uppercase tracking-wide text-white/45">Layers</h2>
        <ul className="flex flex-col gap-1">
          {(p.session?.layers ?? []).map((layer) => (
            <li key={layer.name} className="flex items-center justify-between rounded-md bg-white/5 px-2.5 py-1.5">
              <span className="truncate">{layer.name}</span>
              <button
                type="button"
                data-testid={`lock-${layer.name}`}
                aria-pressed={!!p.lockedLayers[layer.name]}
                onClick={() => p.onLockedLayers({ ...p.lockedLayers, [layer.name]: !p.lockedLayers[layer.name] })}
                className={`rounded px-1.5 py-0.5 text-[11px] ${
                  p.lockedLayers[layer.name] ? "bg-amber-400/20 text-amber-200" : "text-white/45 hover:text-white"
                }`}
              >
                {p.lockedLayers[layer.name] ? "Locked" : "Rotates"}
              </button>
            </li>
          ))}
        </ul>

        <p className="mb-1.5 mt-4 text-xs text-white/45">
          Split a layer with a painted mask (same size as the sketch):
        </p>
        <div className="flex flex-col gap-1.5">
          <button type="button" onClick={() => addMask("plane")} className="rounded-md border border-white/10 px-2.5 py-1.5 text-left text-xs hover:bg-white/5">
            + Glasses / accessory
            <span className="block text-white/40">Floats in front of the face for parallax</span>
          </button>
          <button type="button" onClick={() => addMask("inflate")} className="rounded-md border border-white/10 px-2.5 py-1.5 text-left text-xs hover:bg-white/5">
            + Prop / furniture
            <span className="block text-white/40">Own volume; stays put by default</span>
          </button>
        </div>
        <input ref={fileRef} type="file" accept="image/png,image/webp" className="hidden"
          onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />

        {p.extraLayers.length > 0 && (
          <ul className="mt-2 flex flex-col gap-1">
            {p.extraLayers.map((l) => (
              <li key={l.id} className="flex items-center justify-between rounded-md bg-white/5 px-2.5 py-1 text-xs">
                <span className="truncate">{l.name} <span className="text-white/35">· {l.mask.name}</span></span>
                <button type="button" aria-label={`Remove ${l.name}`} className="text-white/45 hover:text-white"
                  onClick={() => p.onExtraLayers(p.extraLayers.filter((x) => x.id !== l.id))}>×</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {p.dirty && (
        <button type="button" data-testid="rebuild" disabled={p.busy} onClick={p.onRebuild}
          className="rounded-lg bg-white/10 px-3 py-2 text-xs font-medium hover:bg-white/15 disabled:opacity-60">
          Rebuild 3D proxy
        </button>
      )}

      {p.session && p.session.warnings.length > 0 && (
        <section className="rounded-md border border-amber-400/25 bg-amber-400/5 p-2.5 text-xs text-amber-100/85">
          {p.session.warnings.map((w) => (
            <p key={w} className="mb-1 last:mb-0">{w}</p>
          ))}
        </section>
      )}
    </aside>
  );
}
