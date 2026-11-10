import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { dollyDistance, focalFromSlider, focalToFovY, objectEuler, placeCamera, sliderFromFocal } from "@/lib/camera";

describe("dolly zoom", () => {
  it("maps focal length to the full-frame vertical field of view", () => {
    expect((focalToFovY(50) * 180) / Math.PI).toBeCloseTo(26.99, 1);
    expect((focalToFovY(15) * 180) / Math.PI).toBeCloseTo(77.32, 1);
    expect((focalToFovY(200) * 180) / Math.PI).toBeCloseTo(6.87, 1);
  });

  it("keeps the image plane filling the frame at every focal length", () => {
    for (const focal of [15, 24, 50, 85, 200]) {
      const cam = new PerspectiveCamera();
      placeCamera(cam, focal, { viewAspect: 1.5, imageAspect: 1.5, boundsRadius: 2 });
      // Top-right corner of the image plane lands on the corner of the frame.
      const corner = new Vector3(1.5, 1, 0).project(cam);
      expect(corner.x).toBeCloseTo(1, 5);
      expect(corner.y).toBeCloseTo(1, 5);
    }
  });

  it("moves the camera in as the lens gets wider", () => {
    expect(dollyDistance(focalToFovY(15))).toBeCloseTo(1.25, 5);
    expect(dollyDistance(focalToFovY(200))).toBeCloseTo(200 / 12, 5);
  });

  it("exaggerates near geometry at short focal lengths only", () => {
    const size = (focal: number) => {
      const cam = new PerspectiveCamera();
      placeCamera(cam, focal, { viewAspect: 1, imageAspect: 1, boundsRadius: 2 });
      return new Vector3(0, 0.5, 0.5).project(cam).y; // a point half a unit toward the camera
    };
    expect(size(200)).toBeCloseTo(0.5, 1);
    expect(size(15)).toBeGreaterThan(0.8);
  });

  it("widens only the frustum, never the viewpoint, for margins and narrow views", () => {
    const a = new PerspectiveCamera();
    const b = new PerspectiveCamera();
    placeCamera(a, 35, { viewAspect: 1, imageAspect: 1, boundsRadius: 2 });
    placeCamera(b, 35, { viewAspect: 0.5, imageAspect: 1, margin: 1.1, boundsRadius: 2 });
    expect(b.position.z).toBeCloseTo(a.position.z, 9);
    const edge = new Vector3(1, 0, 0).project(b);
    expect(edge.x).toBeCloseTo(1 / 1.1, 5); // image edge sits inside the margin
  });

  it("round-trips the logarithmic focal slider", () => {
    expect(focalFromSlider(0)).toBeCloseTo(15);
    expect(focalFromSlider(1)).toBeCloseTo(200);
    expect(sliderFromFocal(focalFromSlider(0.37))).toBeCloseTo(0.37);
  });

  it("applies roll in the picture plane regardless of yaw", () => {
    const up = new Vector3(0, 1, 0).applyEuler(objectEuler({ yaw: 70, pitch: 0, roll: 90 }));
    expect(up.x).toBeCloseTo(-1, 5);
    expect(up.y).toBeCloseTo(0, 5);
  });
});
