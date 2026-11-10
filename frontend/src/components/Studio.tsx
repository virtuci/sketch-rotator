"use client";

import { AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { absoluteUrl, initSession, refineRotation } from "@/lib/api";
import type { ViewParams } from "@/lib/camera";
import { detectInputMode, imageFromTransfer } from "@/lib/clipboard";
import type { InputMode, ProxyMode, Session } from "@/lib/types";
import type { ViewMode } from "@/shaders/projectiveDisocclusion";
import { BottomBar, DEFAULT_PARAMS } from "./BottomBar";
import { ErrorBoundary } from "./ErrorBoundary";
import { ResultPanel, type LoadedResult } from "./ResultPanel";
import { RotateViewport, type RotateViewportHandle } from "./RotateViewport";
import { SidePanel, type ExtraLayer } from "./SidePanel";

type Busy = { kind: "proxy" | "refine"; message: string; progress: number | null } | null;

export function Studio() {
  const [image, setImage] = useState<{ blob: Blob; url: string } | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>("lineart");
  const [proxyMode, setProxyMode] = useState<ProxyMode>("canonical_anime_head");
  const [extraLayers, setExtraLayers] = useState<ExtraLayer[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [builtWith, setBuiltWith] = useState<string>("");
  const [params, setParams] = useState<ViewParams>(DEFAULT_PARAMS(85));
  const [lockedLayers, setLockedLayers] = useState<Record<string, boolean>>({});
  const [viewMode, setViewMode] = useState<ViewMode>("projected");
  const [showMask, setShowMask] = useState(true);
  const [onionSkin, setOnionSkin] = useState(false);
  const [sideBySide, setSideBySide] = useState(false);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LoadedResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const viewport = useRef<RotateViewportHandle>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);

  const settingsKey = (mode: InputMode, proxy: ProxyMode, layers: ExtraLayer[]) =>
    JSON.stringify([mode, proxy, layers.map((l) => [l.id, l.kind])]);

  const clearResult = useCallback(() => {
    setResult((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev.layerUrl);
        URL.revokeObjectURL(prev.flatUrl);
      }
      return null;
    });
  }, []);

  const buildProxy = useCallback(
    async (blob: Blob, mode: InputMode, proxy: ProxyMode, layers: ExtraLayer[]) => {
      const id = ++requestId.current;
      setBusy({ kind: "proxy", message: "Generating 3D proxy…", progress: null });
      setError(null);
      clearResult();
      try {
        const next = await initSession({
          image: blob,
          inputMode: mode,
          proxyMode: proxy,
          layers: layers.map((l) => ({ spec: { name: l.name, kind: l.kind, locked: l.locked }, mask: l.mask })),
        });
        if (id !== requestId.current) return; // a newer paste superseded this one
        setSession(next);
        setBuiltWith(settingsKey(mode, proxy, layers));
        setParams(DEFAULT_PARAMS(Math.round(next.origin_camera.focal_mm)));
        setLockedLayers(Object.fromEntries(next.layers.map((l) => [l.name, l.locked])));
      } catch (err) {
        if (id !== requestId.current) return;
        setError(err instanceof Error ? err.message : "Could not build the 3D proxy.");
      } finally {
        if (id === requestId.current) setBusy(null);
      }
    },
    [clearResult],
  );

  const loadImage = useCallback(
    async (file: Blob) => {
      setImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { blob: file, url: URL.createObjectURL(file) };
      });
      setSession(null);
      setExtraLayers([]);
      let mode: InputMode = "lineart";
      try {
        mode = await detectInputMode(file);
      } catch {
        setError("That clipboard content is not an image this browser can read.");
        return;
      }
      setInputMode(mode);
      await buildProxy(file, mode, proxyMode, []);
    },
    [buildProxy, proxyMode],
  );

  // Ctrl+V anywhere loads the copied layer.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = imageFromTransfer(e.clipboardData);
      if (file) {
        e.preventDefault();
        void loadImage(file);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [loadImage]);

  const changeInputMode = (mode: InputMode) => {
    if (mode === inputMode) return;
    setInputMode(mode);
    if (image) void buildProxy(image.blob, mode, proxyMode, extraLayers);
  };

  const apply = async () => {
    if (!session || !viewport.current || busy) return;
    setBusy({ kind: "refine", message: "Capturing viewport…", progress: 0 });
    setError(null);
    try {
      const { buffers, camera } = await viewport.current.capture();
      const meta = await refineRotation(session.session_id, buffers, camera, {}, (progress, message) =>
        setBusy({ kind: "refine", message, progress }),
      );
      // Fetch now, so the later copy click can write to the clipboard synchronously.
      const [layer, flat] = await Promise.all(
        [meta.result_rgba_url, meta.result_flat_url].map(async (u) => {
          const res = await fetch(absoluteUrl(u));
          if (!res.ok) throw new Error("The refined image could not be downloaded.");
          return res.blob();
        }),
      );
      clearResult();
      setResult({ meta, layer, flat, layerUrl: URL.createObjectURL(layer), flatUrl: URL.createObjectURL(flat) });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Refinement failed.");
    } finally {
      setBusy(null);
    }
  };

  const dirty = !!session && builtWith !== settingsKey(inputMode, proxyMode, extraLayers);

  return (
    <main
      className="flex h-screen flex-col"
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = imageFromTransfer(e.dataTransfer);
        if (file) void loadImage(file);
      }}
    >
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-white/5 bg-ink-900 px-4">
        <h1 className="text-sm font-semibold tracking-tight">
          TurnSketch <span className="font-normal text-white/45">Studio</span>
        </h1>
        <div className="flex items-center gap-3 text-xs text-white/50">
          {busy && <span role="status" data-testid="status">{busy.message}</span>}
          <button type="button" onClick={() => fileInput.current?.click()} className="rounded-md border border-white/10 px-2.5 py-1 text-white/80 hover:bg-white/5">
            Open image…
          </button>
          <input ref={fileInput} data-testid="file-input" type="file" accept="image/png,image/webp" className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void loadImage(file);
              e.target.value = "";
            }} />
        </div>
      </header>

      {error && (
        <div role="alert" data-testid="error" className="flex items-center justify-between bg-red-500/15 px-4 py-2 text-sm text-red-100">
          <span>{error}</span>
          <button type="button" aria-label="Dismiss" onClick={() => setError(null)} className="text-red-100/70 hover:text-white">×</button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {image && (
          <SidePanel
            session={session}
            proxyMode={proxyMode}
            onProxyMode={setProxyMode}
            extraLayers={extraLayers}
            onExtraLayers={setExtraLayers}
            lockedLayers={lockedLayers}
            onLockedLayers={setLockedLayers}
            dirty={dirty}
            busy={!!busy}
            onRebuild={() => image && void buildProxy(image.blob, inputMode, proxyMode, extraLayers)}
          />
        )}

        <section className={`relative flex min-w-0 flex-1 ${session ? "pb-40" : ""}`}>
          {!image && (
            <div className={`m-6 flex flex-1 flex-col items-center justify-center rounded-2xl border-2 border-dashed text-center transition-colors ${
              dragOver ? "border-accent bg-accent/5" : "border-white/10"
            }`}>
              <p className="text-lg font-medium">Paste a sketch to start</p>
              <p className="mt-2 max-w-md text-sm text-white/50">
                Copy a layer or selection in Clip Studio Paint, Photoshop or Krita, then press
                <kbd className="mx-1 rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs">Ctrl</kbd>+
                <kbd className="mx-1 rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs">V</kbd>
                here. You can also drop a PNG or WebP.
              </p>
              <p className="mt-4 text-xs text-white/35">
                Black lineart on white or transparent, or a cel-shaded frame on #229919.
              </p>
            </div>
          )}

          {image && sideBySide && (
            <div className={`flex w-1/2 items-center justify-center border-r border-white/5 p-6 ${inputMode === "lineart" ? "bg-white" : "checker"}`}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={image.url} alt="Original sketch" className="max-h-full max-w-full object-contain" />
            </div>
          )}

          {image && (
            <div className="relative min-w-0 flex-1 bg-[#2a2d34]">
              {session ? (
                <ErrorBoundary resetKey={session.session_id} onError={setError}>
                  <RotateViewport
                    key={session.session_id}
                    ref={viewport}
                    session={session}
                    params={params}
                    onParamsChange={setParams}
                    lockedLayers={lockedLayers}
                    viewMode={viewMode}
                    showMask={showMask}
                    onionSkin={onionSkin}
                  />
                </ErrorBoundary>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-white/50">
                  {busy?.kind === "proxy" ? busy.message : "No proxy yet."}
                </div>
              )}
            </div>
          )}

          <div className="pointer-events-none absolute inset-x-0 bottom-5">
            <AnimatePresence>
              {session && (
                <BottomBar
                  params={params}
                  originFocal={Math.round(session.origin_camera.focal_mm)}
                  onParams={setParams}
                  inputMode={inputMode}
                  onInputMode={changeInputMode}
                  viewMode={viewMode}
                  onViewMode={setViewMode}
                  showMask={showMask}
                  onShowMask={setShowMask}
                  onionSkin={onionSkin}
                  onOnionSkin={setOnionSkin}
                  sideBySide={sideBySide}
                  onSideBySide={setSideBySide}
                  busy={!!busy}
                  progress={busy?.kind === "refine" ? busy.progress : null}
                  onApply={apply}
                />
              )}
            </AnimatePresence>
          </div>
        </section>

        <AnimatePresence>
          {result && <ResultPanel result={result} inputMode={inputMode} onClose={clearResult} />}
        </AnimatePresence>
      </div>
    </main>
  );
}
