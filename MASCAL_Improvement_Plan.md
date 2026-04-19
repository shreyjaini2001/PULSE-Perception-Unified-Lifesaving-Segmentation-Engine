# MASCAL Triage Relay — Improvement Plan

**Status:** Phases 0–5 scaffolded and running. Core pipeline boots, YOLO detects, dashboard serves.
**Problems identified from screenshots + team feedback:**
1. Camera picks up non-human objects (bottles, books) and rates them as victims.
2. Body scan renders as a flat 2D stick figure in a black box — unusable for wound localization.
3. Wound/injury markers aren't being read from camera and mapped onto the body model.
4. Dashboard UI looks rough and doesn't convey clinical professionalism.
5. Skeleton mapping doesn't accurately note body points or structure.

This document provides step-by-step implementation instructions organized by priority.

---

## Table of contents

1. [Fix 1: Person-only filtering — stop detecting bottles and books](#fix-1-person-only-filtering)
2. [Fix 2: 3D rotatable body model with wound pins](#fix-2-3d-rotatable-body-model)
3. [Fix 3: Accurate skeleton mapping from camera to model](#fix-3-accurate-skeleton-mapping)
4. [Fix 4: Wound detection → body-region mapping pipeline](#fix-4-wound-to-body-mapping)
5. [Fix 5: Dashboard UI overhaul](#fix-5-dashboard-ui-overhaul)
6. [GPU acceleration notes](#gpu-acceleration)
7. [New packages and assets required](#new-packages-and-assets)
8. [Stretch enhancements](#stretch-enhancements)
9. [File-by-file change summary](#file-change-summary)

---

## Fix 1: Person-only filtering

### Problem
YOLOv8-pose detects ALL COCO classes (80 classes: person, bicycle, car, bottle, book, etc.). Your pipeline is treating every detection as a potential victim. The screenshots show "Bravo-2", "Echo-5", "Foxtrot-6" — some of these are likely non-person objects being assigned victim IDs.

### Root cause
In `pipeline/person.py`, the YOLO results are likely not filtered by class ID. YOLOv8-pose returns `cls=0` for "person" but the pose model can still return bounding boxes for other object classes depending on your configuration.

### Fix — step by step

**File: `edge-node/pipeline/person.py`**

1. After running YOLO inference, filter results to class 0 (person) only:

```python
results = self.model(frame, verbose=False)

for result in results:
    boxes = result.boxes
    keypoints = result.keypoints

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])

        # CRITICAL: Only process persons (class 0)
        if cls_id != 0:
            continue

        conf = float(box.conf[0])

        # Also filter by confidence — reject low-confidence person detections
        if conf < 0.5:
            continue

        # Only access keypoints for valid person detections
        kps = keypoints[i] if keypoints is not None else None

        # ... rest of tracking logic
```

2. Add a minimum bounding box area filter to reject tiny spurious detections:

```python
x1, y1, x2, y2 = box.xyxy[0].tolist()
area = (x2 - x1) * (y2 - y1)
frame_area = frame.shape[0] * frame.shape[1]

# Reject detections smaller than 2% of frame — too small to be a real victim
if area < frame_area * 0.02:
    continue
```

3. Add a keypoint validity check — a real person detection should have at least 5 visible keypoints:

```python
if kps is not None:
    visible_kps = sum(1 for k in kps.data[0] if k[2] > 0.3)  # confidence > 0.3
    if visible_kps < 5:
        continue  # Not enough body structure visible — skip
```

### Acceptance test
- Point camera at a desk with bottles, books, and a phone. No victim tiles should appear.
- Walk a person into frame. Exactly one victim tile appears.
- Remove person from frame. Tile shows "last seen X seconds ago" but no new spurious tiles spawn.

---

## Fix 2: 3D rotatable body model

### Problem
The current victim detail view shows a flat 2D stick figure rendered in a small black canvas. This doesn't allow medics to understand where wounds are on the body from different angles.

### Solution
Replace the 2D canvas with an interactive 3D human body model rendered in Three.js, using a pre-made GLTF human model. Wound markers get pinned onto the 3D surface at the correct anatomical location. The medic can click-drag to rotate, scroll to zoom.

### New packages/assets needed

Add to your dashboard (loaded via CDN, no npm needed):

```html
<!-- In dashboard/index.html <head> or before </body> -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
```

You also need a GLTFLoader. Since Three.js r128 on the CDN doesn't bundle addons, use the ESM version:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
```

**3D body model asset — download one of these (free, GLTF format):**

Option A (recommended): Use a simple low-poly anatomical mannequin from CGTrader or Sketchfab. Search for "low poly human body GLTF" — you want something under 2MB, gender-neutral, no textures needed. A semi-transparent mannequin works best for wound visualization.

Option B (fastest, no download): Programmatically build a simplified body from Three.js primitives (capsules, spheres, cylinders). Less realistic but zero external dependencies and guaranteed to work.

**I recommend Option B for the hackathon** because it has zero asset-loading failure modes, renders instantly, and you can customize it completely. Save Option A for a post-hackathon polish pass.

### Implementation — Option B (procedural body)

**Create new file: `dashboard/body3d.js`**

This module exports a function that creates a Three.js scene with a procedural human body and wound pin capability:

```javascript
// dashboard/body3d.js
// 3D rotatable body model with wound pin support

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// COCO keypoint indices for reference
// 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
// 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
// 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
// 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

export function createBodyViewer(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    const width = container.clientWidth || 300;
    const height = container.clientHeight || 400;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);  // dark background

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1.0, 3.0);
    camera.lookAt(0, 0.9, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Orbit controls — click-drag to rotate, scroll to zoom
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.5;
    controls.maxDistance = 6.0;
    controls.update();

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 3, 2);
    scene.add(dirLight);

    // Body material — semi-transparent for wound visibility
    const bodyMat = new THREE.MeshPhongMaterial({
        color: 0x8899aa,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide
    });

    const bodyGroup = new THREE.Group();
    scene.add(bodyGroup);

    // Helper to create limb segments
    function addCapsule(parent, radTop, radBot, height, x, y, z) {
        const geo = new THREE.CylinderGeometry(radTop, radBot, height, 12);
        const mesh = new THREE.Mesh(geo, bodyMat);
        mesh.position.set(x, y, z);
        parent.add(mesh);
        return mesh;
    }

    function addSphere(parent, radius, x, y, z) {
        const geo = new THREE.SphereGeometry(radius, 16, 12);
        const mesh = new THREE.Mesh(geo, bodyMat);
        mesh.position.set(x, y, z);
        parent.add(mesh);
        return mesh;
    }

    // Build procedural body (standing, arms slightly out)
    // All units in meters, origin at feet
    const head = addSphere(bodyGroup, 0.12, 0, 1.65, 0);           // head
    addCapsule(bodyGroup, 0.08, 0.08, 0.10, 0, 1.50, 0);          // neck
    addCapsule(bodyGroup, 0.18, 0.15, 0.45, 0, 1.22, 0);          // torso upper
    addCapsule(bodyGroup, 0.15, 0.14, 0.20, 0, 0.95, 0);          // torso lower (abdomen)

    // Left arm
    addCapsule(bodyGroup, 0.05, 0.045, 0.30, -0.28, 1.30, 0);     // L upper arm
    addCapsule(bodyGroup, 0.04, 0.035, 0.28, -0.35, 1.02, 0);     // L forearm
    addSphere(bodyGroup, 0.04, -0.38, 0.86, 0);                    // L hand

    // Right arm
    addCapsule(bodyGroup, 0.05, 0.045, 0.30, 0.28, 1.30, 0);      // R upper arm
    addCapsule(bodyGroup, 0.04, 0.035, 0.28, 0.35, 1.02, 0);      // R forearm
    addSphere(bodyGroup, 0.04, 0.38, 0.86, 0);                     // R hand

    // Left leg
    addCapsule(bodyGroup, 0.08, 0.06, 0.42, -0.10, 0.64, 0);      // L thigh
    addCapsule(bodyGroup, 0.055, 0.045, 0.40, -0.10, 0.26, 0);    // L shin
    addSphere(bodyGroup, 0.05, -0.10, 0.04, 0.03);                 // L foot

    // Right leg
    addCapsule(bodyGroup, 0.08, 0.06, 0.42, 0.10, 0.64, 0);       // R thigh
    addCapsule(bodyGroup, 0.055, 0.045, 0.40, 0.10, 0.26, 0);     // R shin
    addSphere(bodyGroup, 0.05, 0.10, 0.04, 0.03);                  // R foot

    // Joint spheres for visual connection
    addSphere(bodyGroup, 0.06, -0.22, 1.42, 0);   // L shoulder
    addSphere(bodyGroup, 0.06, 0.22, 1.42, 0);    // R shoulder
    addSphere(bodyGroup, 0.04, -0.10, 0.83, 0);   // L hip
    addSphere(bodyGroup, 0.04, 0.10, 0.83, 0);    // R hip
    addSphere(bodyGroup, 0.045, -0.10, 0.45, 0);  // L knee
    addSphere(bodyGroup, 0.045, 0.10, 0.45, 0);   // R knee

    // Wound pin storage
    const woundPins = [];

    // Body region → 3D coordinate mapping
    const BODY_REGION_COORDS = {
        'head':            { x: 0,     y: 1.65, z: 0.12 },
        'face':            { x: 0,     y: 1.63, z: 0.14 },
        'neck':            { x: 0,     y: 1.50, z: 0.08 },
        'chest':           { x: 0,     y: 1.30, z: 0.16 },
        'upper_torso':     { x: 0,     y: 1.25, z: 0.16 },
        'abdomen':         { x: 0,     y: 0.98, z: 0.14 },
        'lower_torso':     { x: 0,     y: 0.95, z: 0.14 },
        'back':            { x: 0,     y: 1.20, z: -0.16 },
        'left_shoulder':   { x: -0.24, y: 1.42, z: 0.06 },
        'right_shoulder':  { x: 0.24,  y: 1.42, z: 0.06 },
        'left_upper_arm':  { x: -0.30, y: 1.30, z: 0.05 },
        'right_upper_arm': { x: 0.30,  y: 1.30, z: 0.05 },
        'left_forearm':    { x: -0.36, y: 1.05, z: 0.04 },
        'right_forearm':   { x: 0.36,  y: 1.05, z: 0.04 },
        'left_hand':       { x: -0.38, y: 0.86, z: 0.04 },
        'right_hand':      { x: 0.38,  y: 0.86, z: 0.04 },
        'left_thigh':      { x: -0.10, y: 0.65, z: 0.07 },
        'right_thigh':     { x: 0.10,  y: 0.65, z: 0.07 },
        'left_knee':       { x: -0.10, y: 0.45, z: 0.05 },
        'right_knee':      { x: 0.10,  y: 0.45, z: 0.05 },
        'left_shin':       { x: -0.10, y: 0.30, z: 0.04 },
        'right_shin':      { x: 0.10,  y: 0.30, z: 0.04 },
        'left_ankle':      { x: -0.10, y: 0.08, z: 0.03 },
        'right_ankle':     { x: 0.10,  y: 0.08, z: 0.03 },
        'left_foot':       { x: -0.10, y: 0.04, z: 0.06 },
        'right_foot':      { x: 0.10,  y: 0.04, z: 0.06 },
        'groin':           { x: 0,     y: 0.83, z: 0.10 },
        'unknown':         { x: 0,     y: 1.10, z: 0.18 }
    };

    // Wound severity → color
    const SEVERITY_COLORS = {
        'critical':  0xff2222,   // bright red
        'serious':   0xff8800,   // orange
        'moderate':  0xffcc00,   // yellow
        'minor':     0x44cc44,   // green
        'unknown':   0xcccccc    // grey
    };

    /**
     * Add a wound pin to the 3D body
     * @param {string} bodyRegion — key from BODY_REGION_COORDS
     * @param {string} label — e.g. "GSW", "laceration", "burn"
     * @param {string} severity — key from SEVERITY_COLORS
     */
    function addWoundPin(bodyRegion, label, severity = 'unknown') {
        const coords = BODY_REGION_COORDS[bodyRegion] || BODY_REGION_COORDS['unknown'];
        const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS['unknown'];

        // Pin sphere
        const pinGeo = new THREE.SphereGeometry(0.025, 8, 8);
        const pinMat = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.4,
            transparent: false
        });
        const pin = new THREE.Mesh(pinGeo, pinMat);
        pin.position.set(coords.x, coords.y, coords.z);
        bodyGroup.add(pin);

        // Pulsing ring around the pin
        const ringGeo = new THREE.RingGeometry(0.03, 0.045, 16);
        const ringMat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(pin.position);
        ring.lookAt(camera.position);
        bodyGroup.add(ring);

        woundPins.push({ pin, ring, label, bodyRegion, severity });
        return { pin, ring };
    }

    /**
     * Clear all wound pins
     */
    function clearWounds() {
        woundPins.forEach(wp => {
            bodyGroup.remove(wp.pin);
            bodyGroup.remove(wp.ring);
            wp.pin.geometry.dispose();
            wp.ring.geometry.dispose();
        });
        woundPins.length = 0;
    }

    /**
     * Update wounds from victim state
     * @param {Array} wounds — [{body_region, label, severity}, ...]
     */
    function updateWounds(wounds) {
        clearWounds();
        if (!wounds || wounds.length === 0) return;
        wounds.forEach(w => {
            addWoundPin(
                w.body_region || 'unknown',
                w.label || 'wound',
                w.severity || 'unknown'
            );
        });
    }

    // Resize handler
    function onResize() {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    // Animation loop
    let animId;
    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();

        // Make wound rings always face the camera (billboard effect)
        woundPins.forEach(wp => {
            wp.ring.lookAt(camera.position);
            // Gentle pulse
            const scale = 1.0 + 0.15 * Math.sin(Date.now() * 0.004);
            wp.ring.scale.set(scale, scale, scale);
        });

        renderer.render(scene, camera);
    }
    animate();

    // Public API
    return {
        addWoundPin,
        clearWounds,
        updateWounds,
        onResize,
        dispose: () => {
            cancelAnimationFrame(animId);
            renderer.dispose();
            window.removeEventListener('resize', onResize);
        }
    };
}
```

### Integrating into the dashboard

**File: `dashboard/index.html`**

1. Replace the 2D canvas in the victim detail modal with a container div:

```html
<!-- Replace the existing black-box canvas with: -->
<div id="body-3d-container" style="width: 100%; height: 350px; border-radius: 8px; overflow: hidden;"></div>
```

2. Add the import map and module script:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
  import { createBodyViewer } from './body3d.js';
  // Make it globally accessible for the existing app.js
  window.createBodyViewer = createBodyViewer;
</script>
```

**File: `dashboard/app.js`**

3. When opening a victim detail panel, initialize the 3D viewer and pass wounds:

```javascript
// When victim tile is clicked and detail modal opens:
function openVictimDetail(victim) {
    // ... existing modal open logic ...

    // Initialize 3D body viewer
    if (window.bodyViewer) {
        window.bodyViewer.dispose();
    }
    // Small delay to ensure the container is rendered
    setTimeout(() => {
        window.bodyViewer = window.createBodyViewer('body-3d-container');
        if (window.bodyViewer && victim.wounds) {
            window.bodyViewer.updateWounds(victim.wounds);
        }
    }, 100);
}
```

4. On every WebSocket state update, if the detail panel is open for this victim, refresh wounds:

```javascript
// In your WebSocket message handler:
if (currentDetailVictimId === victim.id && window.bodyViewer) {
    window.bodyViewer.updateWounds(victim.wounds);
}
```

---

## Fix 3: Accurate skeleton mapping

### Problem
YOLOv8-pose returns 17 COCO keypoints per person. Currently these are drawn as raw dots/lines in a tiny black canvas without any filtering, normalization, or anatomical interpretation.

### Solution
Map COCO keypoints to anatomical body regions so the wound detection pipeline can say "this wound is near the right thigh" rather than "this wound is at pixel (423, 612)."

**File: `edge-node/pipeline/body_pose.py`**

Replace or augment the existing body pose logic:

```python
# COCO keypoint names (YOLOv8-pose order)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Define body regions from keypoint midpoints
BODY_REGIONS = {
    'head':            lambda kps: midpoint(kps, 'nose', 'nose'),
    'face':            lambda kps: midpoint(kps, 'left_eye', 'right_eye'),
    'neck':            lambda kps: midpoint(kps, 'left_shoulder', 'right_shoulder'),
    'chest':           lambda kps: region_center(kps, ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'], bias_top=0.3),
    'abdomen':         lambda kps: region_center(kps, ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'], bias_top=0.7),
    'left_upper_arm':  lambda kps: midpoint(kps, 'left_shoulder', 'left_elbow'),
    'right_upper_arm': lambda kps: midpoint(kps, 'right_shoulder', 'right_elbow'),
    'left_forearm':    lambda kps: midpoint(kps, 'left_elbow', 'left_wrist'),
    'right_forearm':   lambda kps: midpoint(kps, 'right_elbow', 'right_wrist'),
    'left_hand':       lambda kps: midpoint(kps, 'left_wrist', 'left_wrist'),
    'right_hand':      lambda kps: midpoint(kps, 'right_wrist', 'right_wrist'),
    'left_thigh':      lambda kps: midpoint(kps, 'left_hip', 'left_knee'),
    'right_thigh':     lambda kps: midpoint(kps, 'right_hip', 'right_knee'),
    'left_shin':       lambda kps: midpoint(kps, 'left_knee', 'left_ankle'),
    'right_shin':      lambda kps: midpoint(kps, 'right_knee', 'right_ankle'),
    'left_knee':       lambda kps: midpoint(kps, 'left_knee', 'left_knee'),
    'right_knee':      lambda kps: midpoint(kps, 'right_knee', 'right_knee'),
}


def midpoint(kps_dict, name_a, name_b):
    """Get midpoint between two keypoints. Returns (x, y) or None."""
    a = kps_dict.get(name_a)
    b = kps_dict.get(name_b)
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def region_center(kps_dict, names, bias_top=0.5):
    """Get center of a polygon defined by keypoints, with vertical bias."""
    points = [kps_dict[n] for n in names if n in kps_dict]
    if len(points) < 2:
        return None
    cx = sum(p[0] for p in points) / len(points)
    ys = sorted(p[1] for p in points)
    cy = ys[0] + (ys[-1] - ys[0]) * bias_top
    return (cx, cy)


def keypoints_to_dict(keypoints_tensor, conf_threshold=0.3):
    """Convert YOLOv8 keypoints tensor to named dictionary.

    Args:
        keypoints_tensor: shape (17, 3) — x, y, conf per keypoint
        conf_threshold: minimum confidence to include

    Returns:
        dict of {name: (x, y)} for visible keypoints
    """
    result = {}
    for i, name in enumerate(COCO_KEYPOINTS):
        if i < len(keypoints_tensor):
            x, y, conf = keypoints_tensor[i]
            if conf >= conf_threshold:
                result[name] = (float(x), float(y))
    return result


def compute_body_regions(keypoints_tensor, conf_threshold=0.3):
    """Compute pixel coordinates for all body regions from COCO keypoints.

    Returns:
        dict of {region_name: (x, y)} for computable regions
    """
    kps_dict = keypoints_to_dict(keypoints_tensor, conf_threshold)
    regions = {}
    for region_name, compute_fn in BODY_REGIONS.items():
        try:
            coord = compute_fn(kps_dict)
            if coord is not None:
                regions[region_name] = coord
        except (KeyError, TypeError):
            continue
    return regions


def locate_wound_on_body(wound_center_xy, body_regions):
    """Given a wound's pixel center (x,y), find the nearest body region.

    Args:
        wound_center_xy: (x, y) pixel coordinate of wound center
        body_regions: dict from compute_body_regions()

    Returns:
        (region_name, distance) — closest body region and pixel distance
    """
    if not body_regions:
        return ('unknown', float('inf'))

    wx, wy = wound_center_xy
    best_region = 'unknown'
    best_dist = float('inf')

    for region_name, (rx, ry) in body_regions.items():
        dist = ((wx - rx) ** 2 + (wy - ry) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_region = region_name

    return (best_region, best_dist)
```

### How this connects to the 3D viewer

When `locate_wound_on_body()` returns `"right_thigh"`, the dashboard sends that string to the Three.js body viewer, which looks it up in `BODY_REGION_COORDS` and places a glowing pin at the correct 3D position. The mapping is:

```
Camera frame → YOLO keypoints → body_regions dict → wound_center matched to nearest region → region name sent via WebSocket → Three.js looks up 3D coordinates → pin placed on model
```

---

## Fix 4: Wound detection → body-region mapping

### Problem
Currently `wound.py` finds wound masks/bounding boxes but doesn't associate them with body regions. The 3D viewer needs to know "this laceration is on the left forearm."

### Changes needed

**File: `edge-node/pipeline/wound.py`**

After detecting wounds (via Grounding DINO + SAM or the HSV fallback), compute each wound's center and map it to the nearest body region:

```python
from pipeline.body_pose import compute_body_regions, locate_wound_on_body

def process_wounds(frame, person_bbox, keypoints_tensor, prompts):
    """
    Run wound detection within a person's bounding box,
    then map each wound to a body region.
    """
    # ... existing wound detection logic (Grounding DINO + SAM or HSV) ...
    # Produces: wound_detections = [{ 'label': str, 'bbox': (x1,y1,x2,y2), 'mask': ..., 'confidence': float }, ...]

    # Compute body regions from this person's keypoints
    body_regions = compute_body_regions(keypoints_tensor)

    # Map each wound to a body region
    for wound in wound_detections:
        wx = (wound['bbox'][0] + wound['bbox'][2]) / 2
        wy = (wound['bbox'][1] + wound['bbox'][3]) / 2
        region, distance = locate_wound_on_body((wx, wy), body_regions)
        wound['body_region'] = region
        wound['body_region_distance'] = distance

        # Estimate severity from wound size relative to body
        wound_area = (wound['bbox'][2] - wound['bbox'][0]) * (wound['bbox'][3] - wound['bbox'][1])
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        ratio = wound_area / max(person_area, 1)

        if ratio > 0.05 or wound['label'] in ['amputation', 'exposed bone']:
            wound['severity'] = 'critical'
        elif ratio > 0.02 or wound['label'] in ['blood', 'tourniquet']:
            wound['severity'] = 'serious'
        elif ratio > 0.01:
            wound['severity'] = 'moderate'
        else:
            wound['severity'] = 'minor'

    return wound_detections
```

**File: `state/victim.py`**

Update the victim dataclass to carry structured wound data:

```python
@dataclass
class WoundRegion:
    label: str                    # "laceration", "blood", "burn", etc.
    body_region: str              # "right_thigh", "chest", etc.
    severity: str                 # "critical", "serious", "moderate", "minor"
    confidence: float
    bbox: Tuple[int, int, int, int]
```

**File: `broadcast/ws_server.py`**

Make sure the WebSocket payload includes the wound data with body regions:

```python
# In the victim serialization:
wounds_data = []
for w in victim.wounds:
    wounds_data.append({
        'body_region': w.body_region,
        'label': w.label,
        'severity': w.severity,
        'confidence': w.confidence
    })
# Include in the broadcast payload
victim_payload['wounds'] = wounds_data
```

---

## Fix 5: Dashboard UI overhaul

### Problem
The dashboard looks like a developer prototype, not a clinical tool. Specific issues visible in screenshots:
- Dark theme is fine but text hierarchy is flat — everything is the same visual weight.
- MARCH dots are colored but unlabeled — a second medic doesn't know what M/A/R/C/H means without training.
- Victim tiles are just yellow rectangles with no structure.
- The black box for the body view is jarring and uninformative.
- The SALT legend takes up prime real estate but adds little.

### Changes — priority order

**File: `dashboard/styles.css`**

1. **Victim tile structure** — give each tile clear sections:

```css
.victim-tile {
    border-radius: 12px;
    padding: 16px;
    margin: 8px;
    min-width: 220px;
    max-width: 280px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    cursor: pointer;
    transition: box-shadow 0.2s, transform 0.15s;
    position: relative;
}

.victim-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* SALT color borders instead of full background fill */
.victim-tile[data-salt="immediate"] {
    border-left: 5px solid #e53e3e;
    background: rgba(229, 62, 62, 0.08);
}
.victim-tile[data-salt="delayed"] {
    border-left: 5px solid #ecc94b;
    background: rgba(236, 201, 75, 0.08);
}
.victim-tile[data-salt="minimal"] {
    border-left: 5px solid #48bb78;
    background: rgba(72, 187, 120, 0.08);
}
.victim-tile[data-salt="expectant"] {
    border-left: 5px solid #a0aec0;
    background: rgba(160, 174, 192, 0.08);
}
.victim-tile[data-salt="dead"] {
    border-left: 5px solid #2d3748;
    background: rgba(45, 55, 72, 0.15);
}
```

2. **MARCH dots with tooltips** — add hover labels:

```css
.march-dots {
    display: flex;
    gap: 6px;
    align-items: center;
}
.march-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: white;
    position: relative;
    cursor: help;
}
.march-dot::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 110%;
    left: 50%;
    transform: translateX(-50%);
    background: #1a1a2e;
    color: #e2e8f0;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 400;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
}
.march-dot:hover::after {
    opacity: 1;
}
```

3. **Header bar** — move scenario selector inline, add a subtle pulse for live connection:

```css
.status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: rgba(26, 26, 46, 0.95);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(8px);
}
.connection-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
}
.connection-badge.live {
    background: rgba(72, 187, 120, 0.15);
    color: #48bb78;
}
.connection-badge.live::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #48bb78;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
```

4. **Detail modal** — full-height slide-in panel instead of tiny popup:

```css
.detail-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 420px;
    height: 100vh;
    background: #16162a;
    border-left: 1px solid rgba(255,255,255,0.08);
    overflow-y: auto;
    padding: 24px;
    z-index: 200;
    transform: translateX(100%);
    transition: transform 0.25s ease;
}
.detail-panel.open {
    transform: translateX(0);
}
```

**File: `dashboard/app.js`**

5. **MARCH dot rendering** — add data-tooltip attributes:

```javascript
function renderMarchDots(march) {
    const labels = {
        M: 'Massive hemorrhage',
        A: 'Airway',
        R: 'Respiration',
        C: 'Circulation',
        H: 'Head / Hypothermia'
    };
    const statusColors = {
        normal: '#48bb78',
        unknown: '#a0aec0',
        suspected: '#ecc94b',
        confirmed: '#e53e3e',
        treated: '#4299e1',
        at_risk: '#ecc94b',
        compromised: '#e53e3e',
        managed: '#4299e1',
        clear: '#48bb78',
        distressed: '#e53e3e',
        absent: '#e53e3e',
        stable: '#48bb78',
        shock: '#ecc94b',
        critical: '#e53e3e',
        arrest: '#e53e3e',
    };
    let html = '<div class="march-dots">';
    for (const [key, fullName] of Object.entries(labels)) {
        const status = march[key.toLowerCase()] || march[fullName.toLowerCase().replace(/ \/ /g, '_').replace(/ /g, '_')] || 'unknown';
        const color = statusColors[status] || '#a0aec0';
        html += `<div class="march-dot" style="background:${color}" data-tooltip="${fullName}: ${status}">${key}</div>`;
    }
    html += '</div>';
    return html;
}
```

---

## GPU acceleration

### Notes on GPU changes already made

Since you've switched from CPU to GPU, verify these settings:

**File: `edge-node/pipeline/wound.py`** — SAM device:
```python
# Change from:
device = "cpu"
# To:
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**File: `edge-node/pipeline/person.py`** — YOLO device:
```python
results = self.model(frame, device='0', verbose=False)  # '0' = first GPU
# Or let ultralytics auto-detect:
results = self.model(frame, verbose=False)  # Auto-selects GPU if available
```

**File: `edge-node/main.py`** — add GPU availability check at startup:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Important:** With GPU enabled, SAM 2.1 should go from ~3 Hz to ~15–20 Hz. YOLO should hit 30+ FPS. Grounding DINO should hit ~10 FPS. These are big improvements.

---

## New packages and assets required

### Python packages (add to requirements.txt)

```
# Already present (verify):
ultralytics>=8.2.0          # YOLOv8-pose
torch>=2.2.0                # PyTorch with CUDA
torchvision>=0.17.0
transformers>=4.40.0        # Grounding DINO
opencv-python>=4.9.0
websockets>=12.0
faster-whisper>=1.0.0
llama-cpp-python>=0.2.50
atakcots>=0.4.0
mediapipe>=0.10.9           # Only works on Python ≤3.12
pyyaml>=6.0
scipy>=1.12.0               # For rPPG FFT
numpy>=1.26.0
Pillow>=10.0.0

# Additions needed (if not already present):
supervision>=0.19.0         # Better detection visualization + tracking
```

`supervision` by Roboflow gives you `sv.Detections`, `sv.ByteTrack`, `sv.BoxAnnotator` — much cleaner than hand-rolling tracking. Consider migrating your tracker to this.

### Frontend (CDN, no install)

```html
<!-- Three.js for 3D body model — add to dashboard/index.html -->
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
```

No other frontend packages needed. Keep it vanilla — no React, no build tools.

### New files to create

| File | Purpose |
|---|---|
| `dashboard/body3d.js` | Three.js 3D body viewer module (code provided above) |
| `edge-node/pipeline/body_pose.py` | Replace/augment with body region mapping (code provided above) |

### Files to modify

| File | Change |
|---|---|
| `edge-node/pipeline/person.py` | Add class-0 filter, confidence threshold, min-area filter, min-keypoint check |
| `edge-node/pipeline/wound.py` | Add body-region association for each wound, add severity estimation |
| `edge-node/state/victim.py` | Add `WoundRegion` dataclass with `body_region` and `severity` fields |
| `edge-node/broadcast/ws_server.py` | Include wound body_region + severity in WebSocket payload |
| `dashboard/index.html` | Add Three.js importmap, replace 2D canvas with 3D container |
| `dashboard/app.js` | Initialize 3D viewer on detail open, pass wound data, MARCH tooltip rendering |
| `dashboard/styles.css` | Complete UI overhaul (tile structure, MARCH dots, detail panel, header) |

---

## Stretch enhancements

These are ordered by impact-per-hour. Do them only after the above fixes are confirmed working.

### S1. Pose-matched 3D model (2–3 hours)

Instead of always showing the body in a standing T-pose, rotate the 3D limbs to approximate the actual pose from YOLO keypoints. If the victim is lying on their side, the 3D model should reflect that. This requires computing joint angles from the keypoints and applying rotations to the procedural body segments. This is meaningful because a supine victim looks very different from a standing one, and wound locations on a supine body are interpreted differently.

### S2. Wound region heatmap overlay (1–2 hours)

Instead of just pins, paint a semi-transparent red gradient on the 3D body surface near wound locations. Use Three.js `DecalGeometry` to project a circular gradient onto the body mesh at the wound coordinate. This gives a more intuitive "this whole area is affected" visual.

### S3. Blood pooling volume estimation (2 hours)

Segment the floor/ground around the victim. Detect blood-colored regions outside the person bounding box. Estimate area in pixels, use a rough depth assumption (camera height ÷ person height ratio), convert to approximate square centimeters, estimate volume assuming 2–3mm depth. Display as "<500ml / 500–1500ml / >1500ml" — clinically meaningful ranges. This directly informs MARCH "M" (Massive hemorrhage) assessment.

### S4. Multi-victim scene map (2–3 hours)

Use the camera's movement (frame-to-frame feature matching via OpenCV `cv2.calcOpticalFlowPyrLK` or ORB feature matching) to build a rough 2D layout of the scene. Place victim icons at their relative positions. This is not true SLAM — it's a "poor man's map" that gives the receiver medic spatial awareness. Display it as a minimap in the dashboard sidebar.

### S5. Real GLTF human model (1–2 hours)

Download a free anatomical GLTF model (e.g., from CGTrader or Sketchfab, search "low poly human body GLTF rigged"). Load it via Three.js GLTFLoader instead of the procedural body. This gives a more professional look but introduces an asset dependency (file hosting, loading time, potential format issues). Only do this if the procedural body is confirmed working first.

### S6. Burn percentage calculator (1 hour)

For fire scenarios: when burn wounds are detected, estimate Total Body Surface Area (TBSA) burned using the Rule of Nines. Map detected burn regions to the body zone chart (head = 9%, each arm = 9%, torso front = 18%, torso back = 18%, each leg = 18%, groin = 1%). Display the percentage on the victim tile and MIST card. This is a small code change with high clinical relevance.

### S7. Night/low-light mode (1 hour)

Add a dashboard toggle that switches the color scheme to red-on-black (preserves night vision for medics working in darkness). Military medics use red-light headlamps and their screens should match. Simple CSS variable swap.

### S8. QR code patient tag (30 minutes)

Generate a QR code per victim that encodes their ID + MIST card as a compact JSON. Display it on the dashboard so the medic can screenshot it or print it and physically attach it to the patient. This bridges the digital-physical gap. Use a lightweight JS QR library like `qrcode-generator` (CDN available).

---

## File change summary

### Priority 1 — Must do (fixes the broken stuff)

```
MODIFY  edge-node/pipeline/person.py       — class-0 filter, confidence, area, keypoint checks
MODIFY  edge-node/pipeline/body_pose.py     — body region mapping from keypoints
MODIFY  edge-node/pipeline/wound.py         — associate wounds with body regions + severity
MODIFY  edge-node/state/victim.py           — WoundRegion dataclass
MODIFY  edge-node/broadcast/ws_server.py    — include wound body_region in payload
```

### Priority 2 — Must do (fixes the ugly stuff)

```
CREATE  dashboard/body3d.js                 — Three.js 3D body viewer
MODIFY  dashboard/index.html                — Three.js importmap, 3D container div
MODIFY  dashboard/app.js                    — 3D viewer init, MARCH tooltips, wound rendering
MODIFY  dashboard/styles.css                — full UI overhaul
```

### Priority 3 — Should do (if time permits)

```
MODIFY  edge-node/main.py                  — GPU startup diagnostics
ADD     supervision to requirements.txt     — better tracking
```

### Priority 4 — Stretch

```
S1–S8 as listed above, in that order.
```

---

## Implementation order for Cursor/Claude Code

Feed these instructions to your AI coding assistant in this exact sequence. Each step should be a separate prompt/task:

1. **"In `person.py`, add a filter after YOLO inference: only keep detections with `cls == 0` (person), confidence ≥ 0.5, bounding box area ≥ 2% of frame, and at least 5 visible keypoints. Reject everything else."**

2. **"Rewrite `body_pose.py` to include these functions: `keypoints_to_dict`, `compute_body_regions`, `locate_wound_on_body`. Use the COCO keypoint order. Add a `BODY_REGIONS` dict that computes pixel coordinates for 17+ anatomical regions from keypoint midpoints."** (Paste the code from Fix 3 above.)

3. **"In `wound.py`, after detecting wounds, call `compute_body_regions()` with the person's keypoints, then call `locate_wound_on_body()` for each wound center. Add `body_region` and `severity` fields to each wound dict."**

4. **"Update `victim.py` to add a `WoundRegion` dataclass with fields: `label`, `body_region`, `severity`, `confidence`, `bbox`. Update the `Victim` dataclass to use `List[WoundRegion]`."**

5. **"Update `ws_server.py` to include `body_region`, `label`, and `severity` in the wound entries of the WebSocket broadcast payload."**

6. **"Create `dashboard/body3d.js` — a Three.js ES module that builds a procedural human body from cylinders and spheres, supports OrbitControls for rotation/zoom, and has `addWoundPin(bodyRegion, label, severity)` and `updateWounds(woundsArray)` methods. Use the importmap pattern for Three.js 0.160."** (Paste the full body3d.js code from Fix 2.)

7. **"Update `dashboard/index.html`: add the Three.js importmap, replace the 2D body canvas with a `<div id='body-3d-container'>`, and import body3d.js as a module."**

8. **"Update `dashboard/app.js`: when the victim detail panel opens, initialize the 3D body viewer and pass the victim's wounds. On WebSocket updates, refresh wounds if the panel is open."**

9. **"Overhaul `dashboard/styles.css`: SALT-colored left borders instead of full yellow background, MARCH dots with hover tooltips, sticky header bar with live pulse badge, slide-in detail panel from the right, proper typography hierarchy."**

10. **"Run the pipeline with webcam. Verify: only humans get victim tiles, wounds get body_region labels, 3D model shows wound pins at correct locations, UI looks clean."**

---

## Final notes

- **Test with moulage first, then clean subjects.** The wound pipeline needs visible simulated injuries to prove it works. Clean subjects should get "no wounds detected" — if they don't, your filters are too loose.
- **The 3D body model is the demo wow-moment.** When a judge clicks a victim tile and can rotate a 3D body with glowing wound pins, that's memorable. Invest the time to make it smooth.
- **Don't forget the person filter is the most critical fix.** A system that triages bottles and books is worse than no system at all. Fix this first, before touching the UI.
