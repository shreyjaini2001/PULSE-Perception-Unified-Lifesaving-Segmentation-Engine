/**
 * Three.js 3D body viewer: procedural mannequin, wound pins, optional GLTF,
 * pose-matched lean (S1), decal heatmaps (S2).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DecalGeometry } from "three/addons/geometries/DecalGeometry.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const BODY_REGION_COORDS = {
  head: { x: 0, y: 1.65, z: 0.12 },
  face: { x: 0, y: 1.63, z: 0.14 },
  neck: { x: 0, y: 1.5, z: 0.08 },
  chest: { x: 0, y: 1.3, z: 0.16 },
  upper_torso: { x: 0, y: 1.25, z: 0.16 },
  abdomen: { x: 0, y: 0.98, z: 0.14 },
  lower_torso: { x: 0, y: 0.95, z: 0.14 },
  back: { x: 0, y: 1.2, z: -0.16 },
  left_shoulder: { x: -0.24, y: 1.42, z: 0.06 },
  right_shoulder: { x: 0.24, y: 1.42, z: 0.06 },
  left_upper_arm: { x: -0.3, y: 1.3, z: 0.05 },
  right_upper_arm: { x: 0.3, y: 1.3, z: 0.05 },
  left_forearm: { x: -0.36, y: 1.05, z: 0.04 },
  right_forearm: { x: 0.36, y: 1.05, z: 0.04 },
  left_hand: { x: -0.38, y: 0.86, z: 0.04 },
  right_hand: { x: 0.38, y: 0.86, z: 0.04 },
  left_thigh: { x: -0.1, y: 0.65, z: 0.07 },
  right_thigh: { x: 0.1, y: 0.65, z: 0.07 },
  left_knee: { x: -0.1, y: 0.45, z: 0.05 },
  right_knee: { x: 0.1, y: 0.45, z: 0.05 },
  left_shin: { x: -0.1, y: 0.3, z: 0.04 },
  right_shin: { x: 0.1, y: 0.3, z: 0.04 },
  left_ankle: { x: -0.1, y: 0.08, z: 0.03 },
  right_ankle: { x: 0.1, y: 0.08, z: 0.03 },
  left_foot: { x: -0.1, y: 0.04, z: 0.06 },
  right_foot: { x: 0.1, y: 0.04, z: 0.06 },
  left_torso: { x: -0.08, y: 1.1, z: 0.14 },
  right_torso: { x: 0.08, y: 1.1, z: 0.14 },
  groin: { x: 0, y: 0.83, z: 0.1 },
  left_lower_leg: { x: -0.1, y: 0.35, z: 0.04 },
  right_lower_leg: { x: 0.1, y: 0.35, z: 0.04 },
  torso: { x: 0, y: 1.1, z: 0.14 },
  unknown: { x: 0, y: 1.1, z: 0.18 },
};

// Aligned with DESIGN.md triage palette (see styles.css :root tokens).
const SEVERITY_COLORS = {
  critical: 0xffb3b3, // --p1-primary (soft-tissue pink)
  serious:  0xff9e5e, // --sev-serious
  moderate: 0xfae500, // --p2-fixed
  minor:    0x00e475, // --p3-tertiary
  unknown:  0x8a95a5, // --on-surface-dim
};

// Background + HUD ink
const HUD = {
  bgTop:     0x0a0d11, // --surface
  bgBottom:  0x060809,
  body:      0x2a3242, // dark metallic
  wireframe: 0xc4c7cd, // --on-surface-variant
  gridMajor: 0x2a3242,
  gridMinor: 0x1b2029,
  surface:   0x0a0d11, // for 4px inner pin stroke per DESIGN.md §5
};

function normRegion(id) {
  if (!id) return "unknown";
  return String(id).toLowerCase().replace(/\s+/g, "_");
}

function buildProceduralBody(bodyMat) {
  const bodyGroup = new THREE.Group();
  const meshes = [];

  function addCapsule(parent, radTop, radBot, height, x, y, z) {
    const geo = new THREE.CylinderGeometry(radTop, radBot, height, 12);
    const mesh = new THREE.Mesh(geo, bodyMat);
    mesh.position.set(x, y, z);
    parent.add(mesh);
    meshes.push(mesh);
    return mesh;
  }

  function addSphere(parent, radius, x, y, z) {
    const geo = new THREE.SphereGeometry(radius, 16, 12);
    const mesh = new THREE.Mesh(geo, bodyMat);
    mesh.position.set(x, y, z);
    parent.add(mesh);
    meshes.push(mesh);
    return mesh;
  }

  addSphere(bodyGroup, 0.12, 0, 1.65, 0);
  addCapsule(bodyGroup, 0.08, 0.08, 0.1, 0, 1.5, 0);
  addCapsule(bodyGroup, 0.18, 0.15, 0.45, 0, 1.22, 0);
  addCapsule(bodyGroup, 0.15, 0.14, 0.2, 0, 0.95, 0);
  addCapsule(bodyGroup, 0.05, 0.045, 0.3, -0.28, 1.3, 0);
  addCapsule(bodyGroup, 0.04, 0.035, 0.28, -0.35, 1.02, 0);
  addSphere(bodyGroup, 0.04, -0.38, 0.86, 0);
  addCapsule(bodyGroup, 0.05, 0.045, 0.3, 0.28, 1.3, 0);
  addCapsule(bodyGroup, 0.04, 0.035, 0.28, 0.35, 1.02, 0);
  addSphere(bodyGroup, 0.04, 0.38, 0.86, 0);
  addCapsule(bodyGroup, 0.08, 0.06, 0.42, -0.1, 0.64, 0);
  addCapsule(bodyGroup, 0.055, 0.045, 0.4, -0.1, 0.26, 0);
  addSphere(bodyGroup, 0.05, -0.1, 0.04, 0.03);
  addCapsule(bodyGroup, 0.08, 0.06, 0.42, 0.1, 0.64, 0);
  addCapsule(bodyGroup, 0.055, 0.045, 0.4, 0.1, 0.26, 0);
  addSphere(bodyGroup, 0.05, 0.1, 0.04, 0.03);
  addSphere(bodyGroup, 0.06, -0.22, 1.42, 0);
  addSphere(bodyGroup, 0.06, 0.22, 1.42, 0);
  addSphere(bodyGroup, 0.04, -0.1, 0.83, 0);
  addSphere(bodyGroup, 0.04, 0.1, 0.83, 0);
  addSphere(bodyGroup, 0.045, -0.1, 0.45, 0);
  addSphere(bodyGroup, 0.045, 0.1, 0.45, 0);

  return { bodyGroup, meshes };
}

function nearestMesh(meshes, pos) {
  let best = meshes[0];
  let bestD = Infinity;
  for (const m of meshes) {
    const p = new THREE.Vector3();
    m.getWorldPosition(p);
    const d = p.distanceToSquared(pos);
    if (d < bestD) {
      bestD = d;
      best = m;
    }
  }
  return best;
}

/**
 * @param {string} containerId
 * @param {{ gltfUrl?: string | null, heatmap?: boolean }} options
 */
export function createBodyViewer(containerId, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) return null;

  const gltfUrl = options.gltfUrl ?? null;
  let heatmapEnabled = !!options.heatmap;

  const width = container.clientWidth || 300;
  const height = container.clientHeight || 400;

  const scene = new THREE.Scene();
  // Vertical gradient backdrop via a fullscreen quad — matches the HUD theme.
  const bgTexture = (() => {
    const canvas = document.createElement("canvas");
    canvas.width = 4; canvas.height = 256;
    const ctx = canvas.getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, "#0a0d11");
    grad.addColorStop(1, "#060809");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 4, 256);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  })();
  scene.background = bgTexture;

  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
  camera.position.set(0.15, 1.05, 2.6);
  camera.lookAt(0, 0.95, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.95, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 1.4;
  controls.maxDistance = 5.5;
  controls.autoRotate = false;
  controls.update();

  // HUD-style rim lighting: cool key + warm pink fill from below, no warm
  // yellow — so the pink pins pop.
  scene.add(new THREE.AmbientLight(0x8a95a5, 0.35));
  const key = new THREE.DirectionalLight(0xc4c7cd, 0.7);
  key.position.set(2, 3, 3);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffb3b3, 0.45);
  rim.position.set(-2.2, 1.2, -1.5);
  scene.add(rim);

  // Body: dark metallic base + wireframe overlay clone so the model looks
  // engineered rather than fleshy. Matches the mockup's dark-wireframe HUD.
  const bodyMat = new THREE.MeshStandardMaterial({
    color: HUD.body,
    roughness: 0.65,
    metalness: 0.35,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
  });
  const wireMat = new THREE.MeshBasicMaterial({
    color: HUD.wireframe,
    wireframe: true,
    transparent: true,
    opacity: 0.18,
  });

  // Ground reference grid (subtle)
  const grid = new THREE.GridHelper(3, 12, HUD.gridMajor, HUD.gridMinor);
  grid.material.transparent = true;
  grid.material.opacity = 0.25;
  grid.position.y = 0;
  scene.add(grid);

  let bodyGroup = new THREE.Group();
  let collisionMeshes = [];
  scene.add(bodyGroup);

  const woundPins = [];
  const heatDecals = [];

  function clearHeatmaps() {
    for (const d of heatDecals) {
      bodyGroup.remove(d);
      d.geometry?.dispose?.();
    }
    heatDecals.length = 0;
  }

  function addHeatDecalAt(position, colorHex) {
    if (!heatmapEnabled || !collisionMeshes.length) return;
    const pos = new THREE.Vector3(position.x, position.y, position.z);
    const host = nearestMesh(collisionMeshes, pos);
    host.updateWorldMatrix(true, false);
    const hostPos = new THREE.Vector3();
    host.getWorldPosition(hostPos);
    const localPos = host.worldToLocal(pos.clone());
    const n = pos.clone().sub(hostPos).normalize();
    if (n.lengthSq() < 1e-6) n.set(0, 0, 1);
    const o = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion().setFromUnitVectors(o, n);
    const e = new THREE.Euler().setFromQuaternion(q);
    const size = new THREE.Vector3(0.14, 0.14, 0.12);
    try {
      const geom = new DecalGeometry(host, localPos, e, size);
      const mat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.45,
        depthTest: true,
        polygonOffset: true,
        polygonOffsetFactor: -4,
      });
      const decal = new THREE.Mesh(geom, mat);
      bodyGroup.add(decal);
      heatDecals.push(decal);
    } catch {
      const g = new THREE.SphereGeometry(0.06, 10, 8);
      const mat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.35,
      });
      const s = new THREE.Mesh(g, mat);
      s.position.copy(pos);
      bodyGroup.add(s);
      heatDecals.push(s);
    }
  }

  // Add a translucent wireframe clone on top of every body mesh, so the
  // model reads as a HUD topology scan.
  function addWireOverlayFor(meshes) {
    meshes.forEach((m) => {
      if (!m.geometry) return;
      const overlay = new THREE.Mesh(m.geometry, wireMat);
      overlay.position.copy(m.position);
      overlay.rotation.copy(m.rotation);
      overlay.scale.copy(m.scale).multiplyScalar(1.005);
      m.parent?.add(overlay);
    });
  }

  function setupProcedural() {
    bodyGroup.clear();
    const { bodyGroup: bg, meshes } = buildProceduralBody(bodyMat);
    while (bg.children.length) bodyGroup.add(bg.children[0]);
    collisionMeshes = meshes;
    addWireOverlayFor(meshes);
  }

  function tryLoadGltf(url) {
    if (!url) {
      setupProcedural();
      return;
    }
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        bodyGroup.clear();
        collisionMeshes = [];
        const root = gltf.scene;
        const meshes = [];
        root.traverse((c) => {
          if (c.isMesh) {
            // Swap in our HUD material; keep original geometry.
            c.material = bodyMat;
            collisionMeshes.push(c);
            meshes.push(c);
          }
        });
        root.scale.setScalar(1.2);
        root.position.y = 0;
        bodyGroup.add(root);
        addWireOverlayFor(meshes);
      },
      undefined,
      () => {
        console.warn("[body3d] GLTF load failed, procedural fallback");
        setupProcedural();
      }
    );
  }

  tryLoadGltf(gltfUrl);

  function addWoundPin(bodyRegion, label, severity = "unknown") {
    const rid = normRegion(bodyRegion);
    const coords = BODY_REGION_COORDS[rid] || BODY_REGION_COORDS.unknown;
    const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.unknown;
    const isCritical = severity === "critical" || severity === "serious";

    // DESIGN.md §5 Entry Wound: primary circle with 4px surface inner stroke.
    // We model that as: colored outer disk + inner dark "surface" disk.
    const pinGroup = new THREE.Group();
    pinGroup.position.set(coords.x, coords.y, coords.z);

    const outerGeo = new THREE.CircleGeometry(0.028, 24);
    const outerMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.95,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const outer = new THREE.Mesh(outerGeo, outerMat);
    outer.renderOrder = 10;
    pinGroup.add(outer);

    const innerGeo = new THREE.CircleGeometry(0.012, 20);
    const innerMat = new THREE.MeshBasicMaterial({
      color: HUD.surface,
      transparent: false,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const inner = new THREE.Mesh(innerGeo, innerMat);
    inner.renderOrder = 11;
    inner.position.z = 0.001;
    pinGroup.add(inner);

    // Pulsing halo ring — stronger + larger for critical/serious
    const ringGeo = new THREE.RingGeometry(
      isCritical ? 0.04 : 0.034,
      isCritical ? 0.06 : 0.048,
      24,
    );
    const ringMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: isCritical ? 0.55 : 0.35,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.renderOrder = 9;
    pinGroup.add(ring);

    // Crosshair cross for critical/serious — the "exit wound" treatment.
    let crosshair = null;
    if (isCritical) {
      crosshair = new THREE.Group();
      const chMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 });
      const mkLine = (x1, y1, x2, y2) => {
        const g = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x1, y1, 0.002),
          new THREE.Vector3(x2, y2, 0.002),
        ]);
        return new THREE.Line(g, chMat);
      };
      crosshair.add(mkLine(-0.072, 0, -0.04, 0));
      crosshair.add(mkLine( 0.04, 0,  0.072, 0));
      crosshair.add(mkLine(0, -0.072, 0, -0.04));
      crosshair.add(mkLine(0,  0.04, 0,  0.072));
      crosshair.renderOrder = 12;
      pinGroup.add(crosshair);
    }

    bodyGroup.add(pinGroup);

    woundPins.push({
      group: pinGroup,
      pin: outer,
      inner,
      ring,
      crosshair,
      label,
      bodyRegion: rid,
      severity,
      isCritical,
    });
    if (heatmapEnabled) addHeatDecalAt(coords, color);
    return { pin: outer, ring };
  }

  function clearWounds() {
    clearHeatmaps();
    woundPins.forEach((wp) => {
      if (wp.group) {
        bodyGroup.remove(wp.group);
        wp.group.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          if (o.material && o.material !== bodyMat && o.material !== wireMat) {
            o.material.dispose?.();
          }
        });
      }
    });
    woundPins.length = 0;
  }

  function updateWounds(wounds) {
    clearWounds();
    if (!wounds || !wounds.length) return;
    wounds.forEach((w) => {
      const region = w.body_location || w.body_region || "unknown";
      addWoundPin(region, w.label || "wound", w.severity || "unknown");
    });
  }

  function setHeatmap(on) {
    heatmapEnabled = !!on;
  }

  /** COCO-17 keypoints: [x,y,c] x N — damped lean for demo (S1). */
  function applyPoseFromKeypoints(kp) {
    if (!kp || kp.length < 13) return;
    const ok = (i) => kp[i] && kp[i].length >= 3 && kp[i][2] > 0.25;
    bodyGroup.rotation.set(0, 0, 0);
    if (ok(5) && ok(6)) {
      const dx = kp[6][0] - kp[5][0];
      const dy = kp[6][1] - kp[5][1];
      bodyGroup.rotation.z = Math.atan2(dy, Math.max(Math.abs(dx), 1e-3)) * 0.2;
    }
    if (ok(5) && ok(11)) {
      bodyGroup.rotation.y = Math.atan2(kp[11][0] - kp[5][0], -(kp[11][1] - kp[5][1])) * 0.18;
    }
  }

  function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", onResize);

  let animId;
  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    const t = Date.now() * 0.004;
    woundPins.forEach((wp) => {
      // Billboard the entire pin group at the camera so the flat discs,
      // crosshair and halo always face the viewer.
      if (wp.group) wp.group.lookAt(camera.position);
      const amp = wp.isCritical ? 0.28 : 0.15;
      const sc = 1.0 + amp * Math.sin(t);
      if (wp.ring) {
        wp.ring.scale.set(sc, sc, sc);
        // Critical pins pulse their opacity too.
        if (wp.isCritical && wp.ring.material) {
          wp.ring.material.opacity = 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(t));
        }
      }
    });
    renderer.render(scene, camera);
  }
  animate();

  return {
    addWoundPin,
    clearWounds,
    updateWounds,
    setHeatmap,
    applyPoseFromKeypoints,
    onResize,
    dispose: () => {
      cancelAnimationFrame(animId);
      renderer.dispose();
      window.removeEventListener("resize", onResize);
    },
  };
}
