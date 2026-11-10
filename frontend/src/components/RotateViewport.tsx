"use client";

import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import {
  forwardRef,
  Suspense,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  ClampToEdgeWrapping,
  DepthTexture,
  Group,
  LinearFilter,
  LinearMipmapLinearFilter,
  MathUtils,
  Matrix4,
  Mesh,
  NearestFilter,
  NoColorSpace,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  Texture,
  TextureLoader,
  UnsignedIntType,
  Vector3,
  Vector4,
  WebGLRenderTarget,
  type BufferGeometry,
} from "three";
import { absoluteUrl } from "@/lib/api";
import { placeCamera, objectEuler, type ViewParams } from "@/lib/camera";
import { captureBuffers, type CaptureBuffers } from "@/lib/capture";
import type { CameraMeta, Session } from "@/lib/types";
import {
  createOriginDepthMaterial,
  createProjectiveDisocclusionMaterial,
  VIEW_MODE,
  type ViewMode,
} from "@/shaders/projectiveDisocclusion";

/** Border left around the export frame in the interactive view. */
const PREVIEW_MARGIN = 1.1;
const ORIGIN_DEPTH_SIZE = 2048;

export interface CaptureResult {
  buffers: CaptureBuffers;
  camera: CameraMeta;
}

export interface RotateViewportHandle {
  /** Render the four export buffers at the sketch's resolution for the current pose. */
  capture(): Promise<CaptureResult>;
}

export interface RotateViewportProps {
  session: Session;
  params: ViewParams;
  onParamsChange: (next: ViewParams) => void;
  /** Layer name -> true when the layer should stay put while the rest rotates. */
  lockedLayers: Record<string, boolean>;
  viewMode: ViewMode;
  showMask: boolean;
  /** Draw the original sketch over the export frame. */
  onionSkin: boolean;
}

interface Rig {
  root: Group;
  rotating: Group;
  fixed: Group;
  meshes: { name: string; mesh: Mesh<BufferGeometry, ShaderMaterial> }[];
  materials: ShaderMaterial[];
  boundsRadius: number;
  originCamera: PerspectiveCamera;
  depthScene: Scene;
  depthTarget: WebGLRenderTarget;
  dispose(): void;
}

function buildRig(session: Session, gltfScene: Group, textures: Texture[]): Rig {
  const oc = session.origin_camera;
  const originCamera = new PerspectiveCamera(MathUtils.radToDeg(oc.fov_y), oc.aspect, oc.near, oc.far);
  originCamera.position.set(0, 0, oc.distance);
  originCamera.updateMatrixWorld(true);
  originCamera.updateProjectionMatrix();
  const originViewProj = new Matrix4().multiplyMatrices(
    originCamera.projectionMatrix,
    originCamera.matrixWorldInverse,
  );

  // Shadow map of the proxy as the origin camera sees it.
  const dw = oc.aspect >= 1 ? ORIGIN_DEPTH_SIZE : Math.round(ORIGIN_DEPTH_SIZE * oc.aspect);
  const dh = oc.aspect >= 1 ? Math.round(ORIGIN_DEPTH_SIZE / oc.aspect) : ORIGIN_DEPTH_SIZE;
  const depthTexture = new DepthTexture(dw, dh, UnsignedIntType);
  depthTexture.minFilter = depthTexture.magFilter = NearestFilter;
  const depthTarget = new WebGLRenderTarget(dw, dh, { depthTexture, depthBuffer: true });

  // Lineart: unseen surface is blank paper. Cel: the blurred local colour.
  const disFill = session.input_mode === "lineart" ? new Vector4(1, 1, 1, 1) : new Vector4(0, 0, 0, 0);

  const root = new Group();
  const rotating = new Group();
  const fixed = new Group();
  root.add(rotating, fixed);
  const depthScene = new Scene();
  const meshes: Rig["meshes"] = [];
  let boundsRadius = 0.5;

  session.layers.forEach((layer, i) => {
    const node = gltfScene.getObjectByName(layer.node);
    if (!(node instanceof Mesh)) return;
    const geometry = node.geometry as BufferGeometry;
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (sphere) boundsRadius = Math.max(boundsRadius, sphere.center.length() + sphere.radius);

    const texture = textures[i];
    // Raw sRGB values are passed straight through, so flat cel colours come
    // out of the export bit-exact instead of round-tripping through linear.
    texture.colorSpace = NoColorSpace;
    texture.wrapS = texture.wrapT = ClampToEdgeWrapping;
    texture.minFilter = LinearMipmapLinearFilter;
    texture.magFilter = LinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = 8;
    texture.needsUpdate = true;

    const material = createProjectiveDisocclusionMaterial({
      sketch: texture,
      originDepth: depthTexture,
      originViewProj,
      originCamPos: new Vector3(0, 0, oc.distance),
      originNear: oc.near,
      originFar: oc.far,
      disFill,
    });
    // Blur radius for unseen-surface fill: about 1/16 of the sketch, at any resolution.
    material.uniforms.uDisLod.value = Math.max(0, Math.log2(Math.max(session.width, session.height)) - 4);
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    meshes.push({ name: layer.name, mesh });
    rotating.add(mesh);

    const depthMesh = new Mesh(geometry, createOriginDepthMaterial(texture));
    depthMesh.frustumCulled = false;
    depthScene.add(depthMesh);
  });

  const materials = meshes.map((m) => m.mesh.material);
  return {
    root,
    rotating,
    fixed,
    meshes,
    materials,
    boundsRadius: boundsRadius * 1.05,
    originCamera,
    depthScene,
    depthTarget,
    dispose() {
      materials.forEach((m) => m.dispose());
      depthScene.traverse((o) => o instanceof Mesh && (o.material as ShaderMaterial).dispose());
      depthTarget.dispose();
      depthTexture.dispose();
    },
  };
}

function syncOriginMatrices(rig: Rig) {
  rig.root.updateMatrixWorld(true);
  for (const { mesh } of rig.meshes) {
    const parent = mesh.parent ?? rig.root;
    (mesh.material.uniforms.uOriginFromWorld.value as Matrix4).copy(parent.matrixWorld).invert();
  }
}

interface SceneProps extends Omit<RotateViewportProps, "onParamsChange" | "onionSkin"> {
  captureRef: MutableRefObject<(() => Promise<CaptureResult>) | null>;
}

function ProxyScene({ session, params, lockedLayers, viewMode, showMask, captureRef }: SceneProps) {
  const { gl, scene, camera, size, invalidate } = useThree();
  const gltf = useGLTF(absoluteUrl(session.mesh_url));
  const textures = useLoader(
    TextureLoader,
    session.layers.map((l) => absoluteUrl(l.texture_url)),
  );
  const imageAspect = session.width / session.height;

  const rig = useMemo(() => buildRig(session, gltf.scene, textures), [session, gltf, textures]);
  useEffect(() => () => rig.dispose(), [rig]);

  // The proxy never deforms, so the origin-camera depth is rendered once.
  useLayoutEffect(() => {
    const prev = gl.getRenderTarget();
    gl.setRenderTarget(rig.depthTarget);
    gl.clear();
    gl.render(rig.depthScene, rig.originCamera);
    gl.setRenderTarget(prev);
    invalidate();
  }, [gl, rig, invalidate]);

  useLayoutEffect(() => {
    for (const { name, mesh } of rig.meshes) {
      (lockedLayers[name] ? rig.fixed : rig.rotating).add(mesh);
    }
    invalidate();
  }, [rig, lockedLayers, invalidate]);

  useLayoutEffect(() => {
    rig.rotating.rotation.copy(objectEuler(params));
    invalidate();
  }, [rig, params, viewMode, showMask, invalidate]);

  useFrame(() => {
    const state = placeCamera(camera as PerspectiveCamera, params.focal, {
      viewAspect: size.width / size.height,
      imageAspect,
      margin: PREVIEW_MARGIN,
      boundsRadius: rig.boundsRadius,
    });
    syncOriginMatrices(rig);
    for (const m of rig.materials) {
      m.uniforms.uNear.value = state.near;
      m.uniforms.uFar.value = state.far;
      m.uniforms.uViewMode.value = VIEW_MODE[viewMode];
      m.uniforms.uShowMask.value = showMask;
    }
  });

  useEffect(() => {
    captureRef.current = async () => {
      const exportCamera = new PerspectiveCamera();
      const state = placeCamera(exportCamera, params.focal, {
        viewAspect: imageAspect,
        imageAspect,
        boundsRadius: rig.boundsRadius,
      });
      rig.rotating.rotation.copy(objectEuler(params));
      syncOriginMatrices(rig);
      const buffers = await captureBuffers(gl, scene, exportCamera, rig.materials, session.width, session.height);
      invalidate();
      return {
        buffers,
        camera: {
          yaw_deg: params.yaw,
          pitch_deg: params.pitch,
          roll_deg: params.roll,
          focal_mm: params.focal,
          fov_y: state.fovY,
          distance: state.distance,
          near: state.near,
          far: state.far,
          width: session.width,
          height: session.height,
        },
      };
    };
    return () => {
      captureRef.current = null;
    };
  }, [captureRef, gl, scene, rig, params, session, imageAspect, invalidate]);

  return <primitive object={rig.root} />;
}

function wrapDegrees(v: number): number {
  return ((((v + 180) % 360) + 360) % 360) - 180;
}

export const RotateViewport = forwardRef<RotateViewportHandle, RotateViewportProps>(function RotateViewport(
  { session, params, onParamsChange, lockedLayers, viewMode, showMask, onionSkin },
  ref,
) {
  const captureRef = useRef<(() => Promise<CaptureResult>) | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; start: ViewParams; roll: boolean } | null>(null);
  const [box, setBox] = useState({ w: 1, h: 1 });

  useImperativeHandle(ref, () => ({
    capture() {
      if (!captureRef.current) return Promise.reject(new Error("The 3D proxy is still loading."));
      return captureRef.current();
    },
  }));

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setBox({ w: entry.contentRect.width || 1, h: entry.contentRect.height || 1 }),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Where the export frame falls inside the viewport: the same fit as placeCamera.
  const imageAspect = session.width / session.height;
  const frameH = box.h / (PREVIEW_MARGIN * Math.max(1, imageAspect / (box.w / box.h)));
  const frameW = frameH * imageAspect;

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, start: params, roll: e.shiftKey };
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    const dx = (e.clientX - d.x) * 0.4;
    const dy = (e.clientY - d.y) * 0.4;
    onParamsChange(
      d.roll
        ? { ...d.start, roll: wrapDegrees(d.start.roll - dx) }
        : {
            ...d.start,
            yaw: wrapDegrees(d.start.yaw + dx),
            pitch: MathUtils.clamp(d.start.pitch + dy, -90, 90),
          },
    );
  };
  const endDrag = () => {
    drag.current = null;
  };

  return (
    <div
      ref={wrapRef}
      data-testid="viewport"
      className="relative h-full w-full cursor-grab touch-none select-none active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <Canvas
        frameloop="demand"
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        onCreated={({ gl }) => gl.setClearColor(0x2a2d34, 1)}
      >
        <Suspense fallback={null}>
          <ProxyScene
            session={session}
            params={params}
            lockedLayers={lockedLayers}
            viewMode={viewMode}
            showMask={showMask}
            captureRef={captureRef}
          />
        </Suspense>
      </Canvas>
      <div
        className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 border border-dashed border-white/25"
        style={{ width: frameW, height: frameH }}
      >
        {onionSkin && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={absoluteUrl(session.texture_url)}
            alt=""
            data-testid="onion-skin"
            className="h-full w-full opacity-60"
            style={
              session.input_mode === "lineart"
                ? { filter: "invert(1)", mixBlendMode: "difference" }
                : { opacity: 0.4 }
            }
          />
        )}
      </div>
    </div>
  );
});
