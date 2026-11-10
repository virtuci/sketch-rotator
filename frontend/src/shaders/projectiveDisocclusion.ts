import {
  DoubleSide,
  GLSL3,
  Matrix4,
  NoBlending,
  ShaderMaterial,
  Texture,
  Vector3,
  Vector4,
} from "three";

/**
 * ProjectiveDisocclusionMaterial
 *
 * Projects the sketch from the origin camera (the viewpoint it was drawn
 * from) onto the 3D proxy, and classifies every fragment as seen or unseen by
 * that camera. It writes four outputs at once (MRT):
 *
 *   0  colour   projected sketch (or a debug view in the interactive preview)
 *   1  depth    linear view depth, 16 bits packed into R (high) and G (low)
 *   2  normal   view-space normal * 0.5 + 0.5
 *   3  mask     R disoccluded, G stretch (1 - N.V_orig), B occluded
 *
 * On screen only output 0 is attached; the others are simply dropped.
 *
 * "Origin space" is world space with the object's rotation undone, i.e. the
 * frame in which the origin camera is fixed. Each layer supplies
 * `uOriginFromWorld`, the inverse of its group's world matrix, so rotating
 * layers and locked layers share one shader.
 */

export const VIEW_MODE = { projected: 0, mask: 1, depth: 2, normal: 3 } as const;
export type ViewMode = keyof typeof VIEW_MODE;

const vertexShader = /* glsl */ `
uniform mat4 uOriginFromWorld;
uniform mat4 uOriginViewProj;

out vec4 vOriginClip;
out vec3 vOriginPos;
out vec3 vOriginNormal;
out vec3 vViewNormal;
out float vViewDepth;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vec4 origin = uOriginFromWorld * world;

  vOriginPos = origin.xyz;
  vOriginNormal = mat3(uOriginFromWorld) * mat3(modelMatrix) * normal;
  vOriginClip = uOriginViewProj * origin;

  vec4 view = viewMatrix * world;
  vViewDepth = -view.z;
  vViewNormal = normalMatrix * normal;
  gl_Position = projectionMatrix * view;
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D uSketch;
uniform sampler2D uOriginDepth;
uniform vec3 uOriginCamPos;
uniform float uOriginNear;
uniform float uOriginFar;
uniform float uFacingThreshold;
uniform float uDepthBias;
uniform float uNear;
uniform float uFar;
uniform bool uExport;
uniform int uViewMode;
uniform bool uShowMask;
uniform vec4 uDisFill;   // rgb + weight: flat fill for disoccluded surface
uniform float uDisLod;   // mip level used when the fill weight is below 1

in vec4 vOriginClip;
in vec3 vOriginPos;
in vec3 vOriginNormal;
in vec3 vViewNormal;
in float vViewDepth;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oDepth;
layout(location = 2) out vec4 oNormal;
layout(location = 3) out vec4 oMask;

void main() {
  // --- Projective texture lookup from the origin camera -------------------
  vec2 uv = (vOriginClip.xy / vOriginClip.w) * 0.5 + 0.5;
  bool inFrustum = vOriginClip.w > 0.0 && all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
  vec4 sketch = texture(uSketch, uv);

  // The texture's alpha is the layer's silhouette. The proxy is deliberately
  // a little larger than the drawing, so the drawing decides the outline.
  float alpha = inFrustum ? sketch.a : 0.0;
  if (uExport) {
    // Hard cut here; the capture is supersampled and resolved on the CPU.
    if (alpha < 0.5) discard;
  } else if (alpha < 0.02) {
    discard;
  }

  // --- Was this surface point seen by the origin camera? ------------------
  vec3 toOrigin = normalize(uOriginCamPos - vOriginPos);
  float facing = dot(normalize(vOriginNormal), toOrigin);

  // Shadow-map test: something nearer to the origin camera covers this point.
  float stored = texture(uOriginDepth, uv).r;
  float nearestSeen = uOriginNear * uOriginFar / (uOriginFar - stored * (uOriginFar - uOriginNear));
  float bias = uDepthBias / max(facing, 0.15); // grazing surfaces need more slack
  bool occluded = vOriginClip.w > nearestSeen + bias;

  // A back face on screen means we are looking at the inside of a shell,
  // which the origin camera never saw whatever the normal says.
  bool disoccluded = !gl_FrontFacing || facing <= uFacingThreshold || occluded;
  float stretch = 1.0 - clamp(facing, 0.0, 1.0);

  // --- Colour --------------------------------------------------------------
  // Unseen surface must not show the drawing projected straight through the
  // object. It gets a neutral base for the refiner instead: a flat fill
  // (paper, for lineart) or the heavily blurred local colour (cel).
  vec3 base = mix(textureLod(uSketch, uv, uDisLod).rgb, uDisFill.rgb, uDisFill.a);
  vec3 color = disoccluded ? base : sketch.rgb;

  vec3 n = normalize(vViewNormal) * (gl_FrontFacing ? 1.0 : -1.0);
  float depth01 = clamp((vViewDepth - uNear) / (uFar - uNear), 0.0, 1.0);

  if (uExport) {
    oColor = vec4(color, 1.0);
  } else {
    vec3 shown = color;
    if (uViewMode == 1) {
      shown = vec3(disoccluded ? 1.0 : 0.0, stretch, occluded ? 1.0 : 0.0);
    } else if (uViewMode == 2) {
      shown = vec3(1.0 - depth01);
    } else if (uViewMode == 3) {
      shown = n * 0.5 + 0.5;
    } else if (uShowMask && disoccluded) {
      float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 10.0));
      shown = mix(color, vec3(1.0, 0.25, 0.6), 0.35 + 0.3 * hatch);
    }
    oColor = vec4(shown, alpha); // alpha-to-coverage smooths the outline
  }

  float d16 = floor(depth01 * 65535.0 + 0.5);
  float hi = floor(d16 / 256.0);
  oDepth = vec4(hi / 255.0, (d16 - hi * 256.0) / 255.0, 0.0, 1.0);
  oNormal = vec4(n * 0.5 + 0.5, 1.0);
  oMask = vec4(disoccluded ? 1.0 : 0.0, stretch, occluded ? 1.0 : 0.0, 1.0);
}
`;

export interface ProjectiveUniforms {
  sketch: Texture;
  originDepth: Texture;
  originViewProj: Matrix4;
  originCamPos: Vector3;
  originNear: number;
  originFar: number;
  /** rgb + weight in [0, 1]. Weight 1 = flat fill, 0 = blurred sketch colour. */
  disFill: Vector4;
}

export function createProjectiveDisocclusionMaterial(u: ProjectiveUniforms): ShaderMaterial {
  return new ShaderMaterial({
    name: "ProjectiveDisocclusionMaterial",
    glslVersion: GLSL3,
    vertexShader,
    fragmentShader,
    side: DoubleSide,
    blending: NoBlending,
    alphaToCoverage: true,
    uniforms: {
      uSketch: { value: u.sketch },
      uOriginDepth: { value: u.originDepth },
      uOriginViewProj: { value: u.originViewProj },
      uOriginFromWorld: { value: new Matrix4() },
      uOriginCamPos: { value: u.originCamPos },
      uOriginNear: { value: u.originNear },
      uOriginFar: { value: u.originFar },
      // N.V below this counts as unseen: at 0.2 a texel is already smeared 5x.
      uFacingThreshold: { value: 0.2 },
      uDepthBias: { value: 0.02 },
      uNear: { value: 0.1 },
      uFar: { value: 10 },
      uExport: { value: false },
      uViewMode: { value: VIEW_MODE.projected },
      uShowMask: { value: true },
      uDisFill: { value: u.disFill },
      uDisLod: { value: 5 },
    },
  });
}

/**
 * Depth-only pass rendered once from the origin camera. It uses the same
 * silhouette cut-out as the main pass, so a layer only occludes where it is
 * actually drawn (glasses frames, not the lenses' empty area).
 */
export function createOriginDepthMaterial(sketch: Texture): ShaderMaterial {
  return new ShaderMaterial({
    name: "OriginDepthMaterial",
    glslVersion: GLSL3,
    side: DoubleSide,
    blending: NoBlending,
    uniforms: { uSketch: { value: sketch } },
    vertexShader: /* glsl */ `
      out vec4 vClip;
      void main() {
        vClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = vClip;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      uniform sampler2D uSketch;
      in vec4 vClip;
      layout(location = 0) out vec4 oColor;
      void main() {
        vec2 uv = (vClip.xy / vClip.w) * 0.5 + 0.5;
        if (texture(uSketch, uv).a < 0.5) discard;
        oColor = vec4(1.0);
      }
    `,
  });
}
