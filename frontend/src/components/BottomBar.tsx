"use client";

import { motion } from "framer-motion";
import { FOCAL_MAX_MM, FOCAL_MIN_MM, focalFromSlider, sliderFromFocal, type ViewParams } from "@/lib/camera";
import type { InputMode } from "@/lib/types";
import type { ViewMode } from "@/shaders/projectiveDisocclusion";

export const DEFAULT_PARAMS = (originFocal: number): ViewParams => ({
  yaw: 0,
  pitch: 0,
  roll: 0,
  focal: originFocal,
});

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
  onReset: () => void;
  testId: string;
}

function Slider({ label, value, min, max, step, display, onChange, onReset, testId }: SliderProps) {
  return (
    <label className="flex min-w-[9.5rem] flex-1 flex-col gap-1">
      <span className="flex items-baseline justify-between text-[11px] uppercase tracking-wide text-white/55">
        <button type="button" className="hover:text-white" onClick={onReset} title={`Reset ${label}`}>
          {label}
        </button>
        <span className="font-mono text-xs normal-case text-white/90">{display}</span>
      </span>
      <input
        type="range"
        data-testid={testId}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={onReset}
        className="h-1.5 w-full cursor-pointer"
      />
    </label>
  );
}

function Toggle({ on, onClick, children, testId }: { on: boolean; onClick: () => void; children: React.ReactNode; testId?: string }) {
  return (
    <button
      type="button"
      data-testid={testId}
      aria-pressed={on}
      onClick={onClick}
      className={`rounded-md px-2.5 py-1.5 text-xs transition-colors ${
        on ? "bg-accent/25 text-accent-soft" : "text-white/60 hover:bg-white/5 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

export interface BottomBarProps {
  params: ViewParams;
  originFocal: number;
  onParams: (p: ViewParams) => void;
  inputMode: InputMode;
  onInputMode: (m: InputMode) => void;
  viewMode: ViewMode;
  onViewMode: (m: ViewMode) => void;
  showMask: boolean;
  onShowMask: (v: boolean) => void;
  onionSkin: boolean;
  onOnionSkin: (v: boolean) => void;
  sideBySide: boolean;
  onSideBySide: (v: boolean) => void;
  busy: boolean;
  progress: number | null;
  onApply: () => void;
}

const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: "projected", label: "Sketch" },
  { id: "mask", label: "Mask" },
  { id: "depth", label: "Depth" },
  { id: "normal", label: "Normal" },
];

export function BottomBar(p: BottomBarProps) {
  const set = (patch: Partial<ViewParams>) => p.onParams({ ...p.params, ...patch });
  return (
    <motion.div
      initial={{ y: 40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 40, opacity: 0 }}
      transition={{ type: "spring", stiffness: 260, damping: 26 }}
      className="pointer-events-auto mx-auto flex w-[min(72rem,calc(100vw-2rem))] flex-col gap-3 rounded-2xl border border-white/10 bg-ink-800/90 px-5 py-3.5 shadow-2xl backdrop-blur"
    >
      <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
        <Slider label="Rotate" testId="slider-yaw" value={p.params.yaw} min={-180} max={180} step={1}
          display={`${p.params.yaw.toFixed(0)}°`} onChange={(yaw) => set({ yaw })} onReset={() => set({ yaw: 0 })} />
        <Slider label="Tilt" testId="slider-pitch" value={p.params.pitch} min={-90} max={90} step={1}
          display={`${p.params.pitch.toFixed(0)}°`} onChange={(pitch) => set({ pitch })} onReset={() => set({ pitch: 0 })} />
        <Slider label="Roll" testId="slider-roll" value={p.params.roll} min={-180} max={180} step={1}
          display={`${p.params.roll.toFixed(0)}°`} onChange={(roll) => set({ roll })} onReset={() => set({ roll: 0 })} />
        <Slider label="Perspective" testId="slider-focal" value={sliderFromFocal(p.params.focal)} min={0} max={1} step={0.001}
          display={`${p.params.focal.toFixed(0)}mm`}
          onChange={(t) => set({ focal: Math.round(focalFromSlider(t)) })}
          onReset={() => set({ focal: p.originFocal })} />
        <button
          type="button"
          data-testid="apply"
          disabled={p.busy}
          onClick={p.onApply}
          className="relative h-10 min-w-[9.5rem] overflow-hidden rounded-lg bg-accent px-4 text-sm font-medium text-white transition-opacity disabled:opacity-70"
        >
          {p.busy && p.progress !== null && (
            <span className="absolute inset-y-0 left-0 bg-white/25 transition-[width]" style={{ width: `${p.progress * 100}%` }} />
          )}
          <span className="relative">{p.busy ? "Refining…" : "Apply rotation"}</span>
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-white/5 pt-2.5">
        <div className="flex items-center gap-1 rounded-lg bg-black/25 p-0.5" role="group" aria-label="Input mode">
          <Toggle testId="mode-lineart" on={p.inputMode === "lineart"} onClick={() => p.onInputMode("lineart")}>Anime lineart</Toggle>
          <Toggle testId="mode-cel" on={p.inputMode === "cel_chroma"} onClick={() => p.onInputMode("cel_chroma")}>
            <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-chroma align-middle" />Cel-shaded #229919
          </Toggle>
        </div>
        <div className="flex items-center gap-1" role="group" aria-label="Viewport display">
          {VIEW_MODES.map((m) => (
            <Toggle key={m.id} testId={`view-${m.id}`} on={p.viewMode === m.id} onClick={() => p.onViewMode(m.id)}>{m.label}</Toggle>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <Toggle testId="toggle-mask" on={p.showMask} onClick={() => p.onShowMask(!p.showMask)}>Unseen areas</Toggle>
          <Toggle testId="toggle-onion" on={p.onionSkin} onClick={() => p.onOnionSkin(!p.onionSkin)}>Onion skin</Toggle>
          <Toggle testId="toggle-side" on={p.sideBySide} onClick={() => p.onSideBySide(!p.sideBySide)}>Side by side</Toggle>
          <span className="ml-2 text-[11px] text-white/35">
            {FOCAL_MIN_MM}–{FOCAL_MAX_MM}mm · drag to rotate · shift-drag to roll
          </span>
        </div>
      </div>
    </motion.div>
  );
}
