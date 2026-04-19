/**
 * Simple 2D scene map: victim bboxes as dots + camera_flow drift (S4).
 */

let accX = 0;
let accY = 0;

export function resetMinimap() {
  accX = 0;
  accY = 0;
}

/**
 * @param {HTMLCanvasElement | null} canvas
 * @param {{ victims?: any[]; camera_flow?: { dx?: number; dy?: number }; frame_count?: number }} scene
 */
export function updateMinimap(canvas, scene) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0f141c";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#253044";
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const flow = scene.camera_flow || {};
  const dx = Number(flow.dx) || 0;
  const dy = Number(flow.dy) || 0;
  accX += dx * 0.02;
  accY += dy * 0.02;
  accX *= 0.98;
  accY *= 0.98;

  const victims = scene.victims || [];
  if (!victims.length) {
    ctx.fillStyle = "#8a95a5";
    ctx.font = "11px system-ui";
    ctx.fillText("No victims", 8, h / 2);
    return;
  }

  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const v of victims) {
    const b = v.bbox;
    if (!b || b.length < 4) continue;
    const cx = (b[0] + b[2]) / 2;
    const cy = (b[1] + b[3]) / 2;
    minX = Math.min(minX, cx);
    minY = Math.min(minY, cy);
    maxX = Math.max(maxX, cx);
    maxY = Math.max(maxY, cy);
  }
  if (!isFinite(minX)) return;

  const pad = 40;
  const spanX = Math.max(maxX - minX, 120);
  const spanY = Math.max(maxY - minY, 120);
  const sx = (w - pad * 2) / spanX;
  const sy = (h - pad * 2) / spanY;
  const s = Math.min(sx, sy);

  ctx.save();
  ctx.translate(w / 2 + accX, h / 2 + accY);
  ctx.scale(s, s);
  ctx.translate(-(minX + maxX) / 2, -(minY + maxY) / 2);

  for (const v of victims) {
    const b = v.bbox;
    if (!b || b.length < 4) continue;
    const cx = (b[0] + b[2]) / 2;
    const cy = (b[1] + b[3]) / 2;
    const tag = v.salt_tag || "UNTAGGED";
    const col =
      tag === "RED"
        ? "#e53935"
        : tag === "YELLOW"
          ? "#fbc02d"
          : tag === "GREEN"
            ? "#43a047"
            : tag === "GREY"
              ? "#9e9e9e"
              : tag === "BLACK"
                ? "#212121"
                : "#455a64";
    ctx.beginPath();
    ctx.arc(cx, cy, 8 / s, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.lineWidth = 1 / s;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
  ctx.restore();

  ctx.fillStyle = "#8a95a5";
  ctx.font = "10px system-ui";
  ctx.fillText(`victims ${victims.length} · frame ${scene.frame_count || 0}`, 6, h - 4);
}
