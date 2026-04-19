// MASCAL receiver dashboard — vanilla JS WebSocket client (ES module).

import { updateMinimap } from "./minimap.js";

const WS_URL = (() => {
  const loc = window.location;
  const host = loc.hostname || "localhost";
  return `ws://${host}:8081/`;
})();

const MARCH_LABELS = {
  M: "Massive hemorrhage",
  A: "Airway",
  R: "Respiration",
  C: "Circulation",
  H: "Head / Hypothermia",
};

const CONSCIOUSNESS_LABELS = {
  alert: "Alert",
  voice: "Responds to voice",
  pain: "Responds to pain",
  unresponsive: "Unresponsive",
  unknown: "Unknown",
};

const PRIORITY_COLORS = {
  P1: "#e53935",
  P2: "#ff6d00",
  P3: "#fbc02d",
  P4: "#43a047",
  P5: "#455a64",
};

function humanizeRegion(id) {
  if (id == null || id === "unknown") return "";
  return String(id).replace(/_/g, " ");
}

let ws = null;
let scene = { scenario: "combat_blast", victims: [], frame_count: 0 };
let selectedVictimId = null;
let selectedScanId = null;
let hoveredVictimId = null;
let operatingMode = "scan"; // matches backend default
let currentProfile = "balanced";
let scanSession = { active: false, running: false, target_id: null };
const events = [];
const pendingScans = new Set();
// victim_id -> { step, total, phase } for the head-to-toe sweep UX.
const scanProgress = new Map();

let bodyViewer = null;
let bodyViewerPromise = null;

// ---------- Connection ----------

// Monotonic sequence number of the last event we processed. The server
// stamps ``seq`` on every non-snapshot event; on reconnect we send a
// ``resume`` with this value and the server replays anything newer from
// its ring buffer. Snapshots remain the canonical state, so if the
// replay window is exhausted the next snapshot still restores us.
let lastEventSeq = 0;

function connect() {
  setConn("connecting…", "chip-neutral");
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    setConn("live", "chip-good chip-live");
    // Ask the server to replay any events the old connection missed.
    // Safe even on first connect (lastEventSeq is 0).
    try {
      ws.send(JSON.stringify({ type: "resume", last_seq: lastEventSeq }));
    } catch {}
  };
  ws.onclose = () => {
    setConn("disconnected · retrying", "chip-bad");
    setTimeout(connect, 1500);
  };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (typeof msg.seq === "number" && msg.seq > lastEventSeq) {
      lastEventSeq = msg.seq;
    }
    handleMessage(msg);
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function setConn(text, cls) {
  const el = document.getElementById("conn-state");
  el.textContent = text;
  el.className = "chip " + cls;
}

// ---------- Message handling ----------

function handleMessage(msg) {
  switch (msg.type) {
    case "hello":
      logEvent("Connected to edge node");
      // Seed our seq cursor from the server so the *first* resume after
      // this tab was opened doesn't ask for events the server never
      // buffered (its ring buffer starts at 1, not 0 of some old run).
      if (typeof msg.seq === "number" && msg.seq > lastEventSeq) {
        lastEventSeq = msg.seq;
      }
      break;
    case "resume_ack":
      if ((msg.replayed || 0) > 0) {
        logEvent(`Resumed: ${msg.replayed} event${msg.replayed === 1 ? "" : "s"} replayed`);
      }
      break;
    case "snapshot":
      scene = msg.scene;
      if (msg.mode && msg.mode !== operatingMode) setOperatingMode(msg.mode, false);
      if (msg.profile && msg.profile !== currentProfile) {
        currentProfile = msg.profile;
        document.getElementById("profile-select").value = msg.profile;
      }
      if (msg.scan_session) {
        scanSession = msg.scan_session;
        renderScanSession();
      }
      renderScene();
      break;
    case "transcript":
      document.getElementById("transcript").textContent = msg.text || "—";
      break;
    case "mist": {
      const v = scene.victims.find((x) => x.id === msg.victim_id);
      if (v) v.mist = msg.mist;
      if (selectedVictimId === msg.victim_id) renderDetail();
      logEvent(`MIST card ready for ${msg.victim_id} (${msg.mist?.source || "?"})`);
      break;
    }
    case "scan_ready": {
      const vid = msg.victim_id;
      pendingScans.delete(vid);
      scanProgress.delete(vid);
      const wd = scanWatchdogs.get(vid);
      if (wd) { clearTimeout(wd); scanWatchdogs.delete(vid); }
      const v = scene.victims.find((x) => x.id === vid);
      if (v) {
        v.scans = v.scans || [];
        v.scans.push(msg.scan);
        v.last_scan_id = msg.scan.scan_id;
        if (typeof msg.total_scans === "number") v.total_scan_count = msg.total_scans;
      }
      selectedVictimId = vid;
      selectedScanId = msg.scan.scan_id;
      renderScene();
      renderDetail();
      document.getElementById("victim-detail").classList.remove("hidden");
      const kw = (msg.scan.keywords || []).slice(0, 3).join(", ");
      const idx = msg.scan_index ? ` (scan #${msg.scan_index})` : "";
      logEvent(`Scan captured for ${vid}${idx} · ${msg.scan.priority}${kw ? " · " + kw : ""}`);
      break;
    }
    case "scan_recognized": {
      const vid = msg.victim_id;
      const prior = msg.prior_scan_count || 0;
      showRecognitionToast(vid, prior, "scan");
      logEvent(`Recognized ${vid} from face · adding scan #${prior + 1}`);
      break;
    }
    case "victim_recognized": {
      const vid = msg.victim_id;
      const scans = msg.scan_count || 0;
      if (scans > 0) {
        showRecognitionToast(vid, scans, "reentry");
        logEvent(`${vid} re-entered the scene (${scans} prior scan${scans === 1 ? "" : "s"})`);
      }
      break;
    }
    case "scan_confirmed":
    case "scan_rejected": {
      const vid = msg.victim_id;
      const v = scene.victims.find((x) => x.id === vid);
      if (v && v.scans) {
        const s = v.scans.find((sc) => sc.scan_id === msg.scan_id);
        if (s && typeof msg.wound_idx === "number" && s.wounds && s.wounds[msg.wound_idx]) {
          s.wounds[msg.wound_idx].confirmation =
            msg.type === "scan_confirmed" ? "confirmed" : "rejected";
        }
        if (msg.priority) {
          v.priority = msg.priority;
        }
      }
      renderScene();
      renderDetail();
      break;
    }
    case "scan_progress": {
      const vid = msg.victim_id;
      scanProgress.set(vid, {
        step: msg.step ?? 0,
        total: msg.total ?? 1,
        phase: msg.phase || "sweeping",
      });
      updateScanTileProgress(vid);
      break;
    }
    case "scan_error": {
      const vid = msg.victim_id;
      pendingScans.delete(vid);
      scanProgress.delete(vid);
      const wd = scanWatchdogs.get(vid);
      if (wd) { clearTimeout(wd); scanWatchdogs.delete(vid); }
      logEvent(`Scan failed for ${vid}: ${msg.reason || "unknown"}`);
      renderScene();
      break;
    }
    case "mode":
      setOperatingMode(msg.mode, false);
      break;
    case "scan_session":
      scanSession = {
        active: !!msg.active,
        running: !!msg.running,
        target_id: msg.target_id || null,
      };
      renderScanSession();
      renderScene();
      break;
    case "profile":
      currentProfile = msg.profile;
      document.getElementById("profile-select").value = msg.profile;
      logEvent(`Profile → ${msg.profile}`);
      break;
    case "audit":
      logEvent(describeAudit(msg.event));
      break;
    case "timer_started": {
      const vid = msg.victim_id;
      const kind = (msg.kind || "timer");
      const src = msg.auto ? ` (auto · ${msg.source || "ai"})` : "";
      logEvent(`${kind} timer started for ${vid}${src}`);
      if (msg.auto) {
        showTimerToast(vid, `${kind.toUpperCase()} STARTED`, src.trim() || "auto", "info");
      }
      break;
    }
    case "timer_milestone": {
      const vid = msg.victim_id;
      const tag = (msg.milestone || "").toUpperCase();
      const text = msg.message || `${msg.kind} milestone`;
      logEvent(`${vid} · ${tag} · ${text}`);
      const level = tag === "2H" ? "breach" : tag === "90M" ? "crit" : "warn";
      showTimerToast(vid, `${tag} · ${(msg.kind || "TIMER").toUpperCase()}`, text, level);
      break;
    }
  }
}

function describeAudit(e) {
  if (!e) return "event";
  if (e.kind === "tag_confirmed") return `${e.actor} confirmed ${e.tag} for ${e.victim_id}`;
  if (e.kind === "scan_captured") return `${e.actor} scanned ${e.victim_id}`;
  return JSON.stringify(e);
}

function logEvent(text) {
  const ts = new Date().toLocaleTimeString();
  events.unshift(`${ts} · ${text}`);
  if (events.length > 40) events.length = 40;
  const log = document.getElementById("event-log");
  log.innerHTML = events.map((x) => `<div class="event">${escapeHtml(x)}</div>`).join("");
}

// Brief overlay toast shown when InsightFace recognizes a known victim
// (either re-entry into the scene or a medic starting a rescan).
function showRecognitionToast(vid, priorCount, kind) {
  let host = document.getElementById("recognition-toasts");
  if (!host) {
    host = document.createElement("div");
    host.id = "recognition-toasts";
    host.className = "toast-host";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast toast-recognize toast-${kind || "scan"}`;
  const label =
    kind === "reentry"
      ? `RE-ENTRY · ${escapeHtml(vid)}`
      : `RECOGNIZED · ${escapeHtml(vid)}`;
  const sub =
    kind === "reentry"
      ? `${priorCount} prior scan${priorCount === 1 ? "" : "s"} restored`
      : `Appending scan #${priorCount + 1}`;
  el.innerHTML = `
    <div class="toast-icon">⌁</div>
    <div class="toast-body">
      <div class="toast-title">${label}</div>
      <div class="toast-sub">${escapeHtml(sub)}</div>
    </div>`;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 400);
  }, 3800);
}

// Timer lifecycle + TCCC milestone notifications (60m / 90m / 2h).
function showTimerToast(vid, title, sub, level) {
  let host = document.getElementById("recognition-toasts");
  if (!host) {
    host = document.createElement("div");
    host.id = "recognition-toasts";
    host.className = "toast-host";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast toast-timer toast-timer-${level || "warn"}`;
  el.innerHTML = `
    <div class="toast-icon">◷</div>
    <div class="toast-body">
      <div class="toast-title">${escapeHtml(title)} · ${escapeHtml(vid || "")}</div>
      <div class="toast-sub">${escapeHtml(sub || "")}</div>
    </div>`;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  // Breach alerts stick around longer — a medic who walks past shouldn't
  // miss the 2-hour hard stop.
  const lifetime = level === "breach" ? 7000 : 4500;
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 400);
  }, lifetime);
}

// Called by thumbs up / thumbs down buttons on individual wound legend rows.
function confirmWound(vid, scanId, woundIdx, decision) {
  const msg = {
    type: decision === "confirmed" ? "confirm_wound" : "reject_wound",
    victim_id: vid,
    scan_id: scanId,
    wound_idx: Number(woundIdx),
    actor: "medic",
  };
  send(msg);
  logEvent(
    `${decision === "confirmed" ? "Confirmed" : "Rejected"} wound #${
      woundIdx + 1
    } for ${vid}`,
  );
}

// ---------- Operating mode + profile ----------

function setOperatingMode(mode, sendToServer = true) {
  operatingMode = mode;
  const pill = document.getElementById("mode-pill");
  pill.querySelectorAll(".pill-opt").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
  document.body.classList.toggle("mode-live", mode === "live");
  if (sendToServer) send({ type: "set_mode", mode, actor: "medic" });
}

function renderScanSession() {
  const btn = document.getElementById("scan-session-btn");
  const chip = document.getElementById("scan-session-state");
  if (!btn || !chip) return;
  if (scanSession.active) {
    btn.textContent = "Stop scan";
    btn.classList.add("active");
    chip.className = `chip ${scanSession.running ? "chip-good chip-live" : "chip-neutral"}`;
    chip.textContent = scanSession.running
      ? `scanning ${scanSession.target_id || ""}`.trim()
      : "waiting for target";
  } else {
    btn.textContent = "Start scan";
    btn.classList.remove("active");
    chip.className = "chip chip-neutral";
    chip.textContent = "idle";
  }
}

function showConfirmModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.remove("hidden");
}
function hideConfirmModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.add("hidden");
}

// ---------- 3D body viewer ----------

function disposeBodyViewer() {
  if (bodyViewer?.dispose) bodyViewer.dispose();
  bodyViewer = null;
  bodyViewerPromise = null;
}

async function syncBodyViewer(v, wounds, keypoints) {
  const el = document.getElementById("body-3d-container");
  if (!el || !v) return;

  const heatOn = document.getElementById("heatmap-toggle")?.checked ?? false;
  const tryGltf = "assets/human.glb";

  if (!window.WebGLRenderingContext) {
    el.innerHTML = `<p class="hint">WebGL unavailable — 2D graph below.</p>`;
    return;
  }

  try {
    if (!bodyViewerPromise) {
      bodyViewerPromise = import("./body3d.js");
    }
    const mod = await bodyViewerPromise;
    disposeBodyViewer();
    let gltfUrl = null;
    try {
      const head = await fetch(tryGltf, { method: "HEAD" });
      if (head.ok) gltfUrl = tryGltf;
    } catch {
      gltfUrl = null;
    }
    bodyViewer = mod.createBodyViewer("body-3d-container", { gltfUrl, heatmap: heatOn });
    if (bodyViewer) {
      bodyViewer.updateWounds(wounds || []);
      bodyViewer.applyPoseFromKeypoints(keypoints || []);
    }
  } catch (e) {
    console.warn("[dashboard] 3D viewer failed", e);
    el.innerHTML = `<p class="hint">3D viewer failed to load — using 2D graph.</p>`;
  }
}

// ---------- Rendering ----------

function renderScene() {
  document.getElementById("scenario-select").value = scene.scenario;
  document.getElementById("victim-count").textContent = scene.victims.length;
  document.getElementById("frame-count").textContent = scene.frame_count || 0;

  // Priority matrix counts
  const counts = { P1: 0, P2: 0, P3: 0, P4: 0, P5: 0 };
  for (const v of scene.victims) {
    const p = v.priority || "P5";
    if (counts[p] != null) counts[p] += 1;
  }
  for (const p of Object.keys(counts)) {
    const el = document.getElementById(`p-count-${p}`);
    if (el) el.textContent = counts[p];
  }

  const grid = document.getElementById("victim-grid");
  const empty = document.getElementById("empty-state");
  const sorted = [...scene.victims].sort((a, b) => sortKey(a) - sortKey(b));
  grid.innerHTML = sorted.map(tileHtml).join("");
  empty.style.display = sorted.length ? "none" : "block";

  grid.querySelectorAll(".tile").forEach((el) => {
    const vid = el.dataset.vid;
    el.classList.toggle("is-hovered", hoveredVictimId === vid);
    el.addEventListener("mouseenter", () => {
      hoveredVictimId = vid;
      el.classList.add("is-hovered");
    });
    el.addEventListener("mouseleave", () => {
      if (hoveredVictimId === vid) hoveredVictimId = null;
      el.classList.remove("is-hovered");
    });
    el.addEventListener("click", (e) => {
      if (e.target.closest("[data-action]")) return;
      openDetail(vid);
    });
  });
  grid.querySelectorAll("[data-action='scan']").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      triggerScan(btn.dataset.vid);
    });
  });
  grid.querySelectorAll("[data-action='confirm-tag']").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const vid = btn.dataset.vid;
      const tag = btn.dataset.tag;
      if (!vid || !tag) return;
      confirmTag(vid, tag);
    });
  });
  grid.querySelectorAll("[data-action='handoff']").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const vid = btn.dataset.vid;
      if (!vid) return;
      playHandoff(vid);
      // Quick visual pulse so the medic knows the tap landed even
      // before the TTS kicks in (browser voice load can take 200ms).
      btn.classList.add("playing");
      setTimeout(() => btn.classList.remove("playing"), 600);
    });
  });

  updateMinimap(document.getElementById("minimap-canvas"), scene);

  if (selectedVictimId) {
    const v = scene.victims.find((x) => x.id === selectedVictimId);
    if (v) {
      renderDetail();
      // Live re-sync 3D when in LIVE mode; in SCAN we only refresh on demand.
      if (operatingMode === "live" && bodyViewer) {
        bodyViewer.updateWounds(v.wound_regions || []);
        bodyViewer.applyPoseFromKeypoints(v.keypoints || []);
      }
    }
  }
}

const TAG_ORDER = ["RED", "YELLOW", "GREEN", "GREY", "BLACK", "UNTAGGED"];
const PRIORITY_ORDER = ["P1", "P2", "P3", "P4", "P5"];
function sortKey(v) {
  const scanned = (v.total_scan_count || (v.scans || []).length || 0) > 0 ? 0 : 1;
  const confirmed = v.salt_tag_confirmed ? 0 : 1;
  const away = v.off_screen ? 1 : 0;
  const p = PRIORITY_ORDER.indexOf(v.priority || "P5");
  const tag = TAG_ORDER.indexOf(v.salt_tag);
  const numericId = Number.parseInt((v.id || "").split(/[_-]/).pop(), 10);
  const idOrder = Number.isFinite(numericId) ? numericId : (v.id.codePointAt(0) || 0);
  return scanned * 100000 + confirmed * 20000 + p * 1000 + away * 100 + tag * 10 + idOrder;
}

function floorBloodSummary(v) {
  const pools = (v.blood_regions || []).filter((b) => b.is_floor_pool);
  if (!pools.length) return "";
  const worst = pools.reduce((a, b) => {
    const rank = { "<500ml": 1, "500-1500ml": 2, ">1500ml": 3 };
    return (rank[b.volume_bucket] || 0) > (rank[a.volume_bucket] || 0) ? b : a;
  }, pools[0]);
  return ` · floor blood ~${worst.volume_bucket || "?"}`;
}

function priorityChip(v) {
  const p = v.priority || "P5";
  const color = PRIORITY_COLORS[p] || "#455a64";
  return `<span class="priority-chip" style="background:${color}">${p}</span>`;
}

function faceAvatarHtml(v, size = 36) {
  const url = v.face_thumb_url || (v.last_scan_id ? `/api/scans/${v.last_scan_id}/face.jpg` : null);
  const initial = (v.id || "?").charAt(0);
  const sz = `${size}px`;
  // Slot keeps fixed size so snapshots don't collapse the layout. Image
  // errors toggle ``show-fallback`` instead of ``onerror`` removal (which
  // caused a load/remove/load flicker loop on 404s).
  if (url) {
    const safeUrl = escapeHtml(url);
    return `<div class="face-avatar-slot" style="width:${sz};height:${sz}">
      <img class="face-avatar" width="${size}" height="${size}" src="${safeUrl}" alt=""
           loading="lazy" decoding="async"
           onerror="this.classList.add('face-avatar-broken'); this.closest('.face-avatar-slot')?.classList.add('show-fallback');"/>
      <div class="face-avatar-fallback face-avatar-fallback-under" aria-hidden="true">${escapeHtml(initial)}</div>
    </div>`;
  }
  return `<div class="face-avatar-slot show-fallback" style="width:${sz};height:${sz}">
    <div class="face-avatar face-avatar-fallback">${escapeHtml(initial)}</div>
  </div>`;
}

function tileHtml(v) {
  const march = v.march || {};
  const dots = ["M", "A", "R", "C", "H"]
    .map((k) => {
      const f = march[k] || { status: "unknown", reason: "" };
      const tip = `${MARCH_LABELS[k]}: ${f.status}${f.reason ? " — " + f.reason : ""}`;
      return `<div class="march-dot" data-status="${f.status}" data-tooltip="${escapeHtml(tip)}">${k}</div>`;
    })
    .join("");

  const vitals = v.vitals || {};
  const hrVal = vitals.hr != null ? Math.round(vitals.hr) : null;
  const rrVal = vitals.rr != null ? Math.round(vitals.rr) : null;
  const hrClass = hrVal == null ? "" : hrVal >= 120 || hrVal <= 50 ? "crit" : hrVal >= 100 ? "warn" : "";
  const rrClass = rrVal == null ? "" : rrVal >= 28 || rrVal <= 8 ? "crit" : rrVal >= 22 ? "warn" : "";
  pushVitalSample(v.id, "hr", hrVal);
  pushVitalSample(v.id, "rr", rrVal);
  const hr = `
    <div class="v">
      <span class="lbl">HR</span>
      <b class="${hrClass}">${hrVal != null ? hrVal : "—"}</b>
      ${sparklineSvg(v.id, "hr", hrClass)}
    </div>`;
  const rr = `
    <div class="v">
      <span class="lbl">RR</span>
      <b class="${rrClass}">${rrVal != null ? rrVal : "—"}</b>
      ${sparklineSvg(v.id, "rr", rrClass)}
    </div>`;

  const tbsa =
    v.tbsa_burn_percent != null
      ? `<div class="v"><span class="lbl">TBSA</span><b class="warn">${v.tbsa_burn_percent}%</b></div>`
      : "";

  const wounds = (v.wound_regions || [])
    .slice(0, 4)
    .map((w) => {
      const loc = w.body_location && w.body_location !== "unknown" ? humanizeRegion(w.body_location) : "";
      const sev = w.severity && w.severity !== "unknown" ? ` (${w.severity})` : "";
      return `<span class="wound-badge">${escapeHtml(w.label)}${loc ? " · " + escapeHtml(loc) : ""}${escapeHtml(sev)}</span>`;
    })
    .join("");

  const timers = (v.timers || [])
    .map((t) => {
      const total = t.duration_seconds || 7200;
      const elapsed = t.elapsed_seconds || 0;
      const remaining = Math.max(0, total - elapsed);
      // TCCC tourniquet milestones — the 2h mark is a hard clinical
      // threshold; a TQ past 2h requires vascular assessment and
      // conversion consideration.
      let cls = "ok";
      let milestone = "";
      const isTq = (t.kind || "").toLowerCase().startsWith("tourniquet");
      if (isTq) {
        if (elapsed >= 7200) { cls = "breach"; milestone = "2H · convert"; }
        else if (elapsed >= 5400) { cls = "crit"; milestone = "90M · prep convert"; }
        else if (elapsed >= 3600) { cls = "warn"; milestone = "60M · reassess"; }
      } else {
        const pct = elapsed / total;
        if (pct > 0.9) cls = "crit";
        else if (pct > 0.75) cls = "warn";
      }
      const milestoneEl = milestone
        ? `<span class="timer-milestone">${escapeHtml(milestone)}</span>`
        : "";
      const autoTag = t.auto ? `<span class="timer-auto">AUTO</span>` : "";
      return `<div class="timer-chip ${cls}" data-kind="${escapeHtml(t.kind)}">`
        + `${autoTag}<span class="timer-kind">${escapeHtml(t.kind)}</span>`
        + ` · <span class="timer-clock">${formatClock(remaining)}</span>`
        + `${milestoneEl}</div>`;
    })
    .join("");

  const codes = (v.tccc_codewords || [])
    .slice(0, 4)
    .map((c) => `<span class="codeword-chip">${escapeHtml(c)}</span>`)
    .join("");

  const scanCount = (v.scans || []).length;
  const totalScanCount = v.total_scan_count || scanCount;
  const isAway = !!v.off_screen;
  const isScanning = pendingScans.has(v.id);
  const prog = scanProgress.get(v.id);
  const pct = prog && prog.total > 0 ? Math.min(100, Math.round((prog.step / prog.total) * 100)) : 0;
  const phaseLabel = {
    starting: "Starting…",
    sweeping: "Sweeping body",
    analyzing: "Analyzing",
    face: "Matching face",
    finalizing: "Finalizing",
  }[prog?.phase || "sweeping"] || "Scanning…";
  const scanBtnLabel = isScanning
    ? `${phaseLabel}${prog ? ` · ${pct}%` : ""}`
    : (scanCount ? `Rescan (${scanCount})` : "Start scan");
  const scanBtnClass = isScanning ? "scan-btn pending" : "scan-btn";
  const scanBtnDisabled = (isScanning || scanSession.active) ? "disabled" : "";
  const autoBadge = scanSession.running && scanSession.target_id === v.id
    ? `<span class="away-pill" title="Current auto-scan target">AUTO</span>`
    : "";
  const sweepOverlay = isScanning
    ? `<div class="scan-sweep">
         <div class="scan-sweep-line"></div>
         <div class="scan-rec-pill" aria-label="Recording audio"><span class="rec-dot"></span>REC AUDIO</div>
         <div class="scan-sweep-label">${escapeHtml(phaseLabel)}</div>
       </div>`
    : "";

  const confirmed = v.salt_tag_confirmed
    ? ""
    : `<div class="scan-pulse" title="AI suggestion awaiting medic confirmation"></div>`;

  const priority = v.priority || "P5";
  const shortId = (v.id || "").split(/[_-]/).pop().slice(0, 6).toUpperCase();

  const tileClass = [
    "tile",
    isScanning ? "scanning" : "",
    hoveredVictimId === v.id ? "is-hovered" : "",
    isAway ? "away" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const awayPill = isAway
    ? `<span class="away-pill" title="Off camera — last seen ${new Date((v.last_on_screen || v.last_seen || 0) * 1000).toLocaleTimeString()}">AWAY</span>`
    : "";
  const recordPill = totalScanCount > 0
    ? `<span class="away-pill" title="${totalScanCount} stored scan record${totalScanCount === 1 ? "" : "s"}">REC</span>`
    : `<span class="away-pill" title="Live detection only — no frozen scan record yet">LIVE</span>`;
  const scanBadge = totalScanCount > 0
    ? `<span class="scan-count-badge" title="${totalScanCount} scan${totalScanCount === 1 ? "" : "s"} in record">${totalScanCount}</span>`
    : "";

  return `
    <div class="${tileClass}" data-vid="${escapeHtml(v.id)}" data-p="${priority}">
      ${confirmed}
      ${sweepOverlay}
      <div class="rail">
        <span class="rail-vtid">${escapeHtml(shortId)}</span>
        <span class="rail-p">${priority}</span>
        <span class="rail-tag">${v.salt_tag}</span>
      </div>
      <div class="tile-body">
        <div class="tile-head">
          <div class="tile-head-left">
            ${faceAvatarHtml(v, 40)}
            <div>
              <div class="tile-id">${escapeHtml(v.id)}${scanBadge}${recordPill}${autoBadge}${awayPill}</div>
              <div class="tile-subid"><span class="tag-chip tag-${v.salt_tag}">${v.salt_tag}</span> ${v.salt_tag_confirmed ? "<span class='muted small'>CONFIRMED</span>" : "<span class='muted small'>PROVISIONAL</span>"}</div>
            </div>
          </div>
          <div class="tile-head-actions">
            <button type="button"
                    class="tile-handoff-btn"
                    data-action="handoff"
                    data-vid="${escapeHtml(v.id)}"
                    title="Play 5-second voice handoff"
                    aria-label="Play voice handoff">◐</button>
            <button type="button" class="${scanBtnClass}" data-action="scan" data-vid="${escapeHtml(v.id)}" ${scanBtnDisabled}>${scanBtnLabel}</button>
          </div>
        </div>
        <div class="tile-reason">${escapeHtml(v.salt_tag_reason || "—")}</div>
        <div class="march-row">${dots}</div>
        <div class="vitals-row">${hr}${rr}${tbsa}</div>
        <div class="tile-wounds">${wounds || "<span class='muted small'>no wounds scanned yet</span>"}<span class="muted small">${escapeHtml(floorBloodSummary(v))}</span></div>
        ${codes ? `<div class="codewords-row">${codes}</div>` : ""}
        ${timers}
        ${tagConfirmStrip(v)}
      </div>
    </div>`;
}

// One-tap SALT confirmation strip that sits at the bottom of an
// unconfirmed tile.  The suggested tag reads as the primary "CONFIRM"
// CTA; the remaining four SALT colors are offered as discreet override
// buttons so the medic can retag without opening the detail panel.
// Dead/Expectant still require the detail panel (policy: never one-tap
// GREY or BLACK — matches the hard gate in state/salt.py).
const TAG_ONE_TAP = ["RED", "YELLOW", "GREEN"];
function tagConfirmStrip(v) {
  if (v.salt_tag_confirmed) return "";
  const suggested = v.salt_tag && TAG_ONE_TAP.includes(v.salt_tag) ? v.salt_tag : null;
  const others = TAG_ONE_TAP.filter((t) => t !== suggested);
  const primary = suggested
    ? `<button type="button"
                class="tag-confirm-primary tag-${suggested}"
                data-action="confirm-tag"
                data-vid="${escapeHtml(v.id)}"
                data-tag="${suggested}">
         Confirm ${suggested}
       </button>`
    : `<span class="muted small tag-confirm-hint">Tap to tag</span>`;
  const overrides = others
    .map(
      (t) =>
        `<button type="button"
                 class="tag-confirm-alt tag-${t}"
                 data-action="confirm-tag"
                 data-vid="${escapeHtml(v.id)}"
                 data-tag="${t}"
                 title="Override to ${t}">${t.slice(0, 1)}</button>`
    )
    .join("");
  return `
    <div class="tag-confirm-strip" data-vid="${escapeHtml(v.id)}">
      ${primary}
      <div class="tag-confirm-overrides">${overrides}</div>
    </div>`;
}

// --- Vital sparklines (kinetic area chart, no axes) ---

const VITAL_HISTORY = new Map(); // vid -> { hr: number[], rr: number[] }
const VITAL_WINDOW = 24;

function pushVitalSample(vid, key, val) {
  if (val == null || Number.isNaN(val)) return;
  const bucket = VITAL_HISTORY.get(vid) || { hr: [], rr: [] };
  const arr = bucket[key] || (bucket[key] = []);
  arr.push(val);
  if (arr.length > VITAL_WINDOW) arr.shift();
  VITAL_HISTORY.set(vid, bucket);
}

function sparklineSvg(vid, key, statusClass) {
  const bucket = VITAL_HISTORY.get(vid) || {};
  const series = bucket[key] || [];
  if (series.length < 2) {
    return `<svg class="sparkline ${statusClass === "crit" ? "critical" : statusClass === "warn" ? "warn" : "stable"}" viewBox="0 0 100 32" preserveAspectRatio="none"></svg>`;
  }
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = Math.max(1, max - min);
  const step = 100 / (series.length - 1);
  const pts = series.map((v, i) => [i * step, 32 - ((v - min) / span) * 24 - 4]);
  const line = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
  const area = line + ` L ${pts[pts.length-1][0].toFixed(1)},32 L ${pts[0][0].toFixed(1)},32 Z`;
  const cls = statusClass === "crit" ? "critical" : statusClass === "warn" ? "warn" : "stable";
  return `<svg class="sparkline ${cls}" viewBox="0 0 100 32" preserveAspectRatio="none">
    <path class="area" d="${area}"/>
    <path class="line" d="${line}"/>
  </svg>`;
}

const scanWatchdogs = new Map();
const SCAN_WATCHDOG_MS = 25000;

function triggerScan(vid) {
  if (pendingScans.has(vid)) return;
  pendingScans.add(vid);
  scanProgress.set(vid, { step: 0, total: 3, phase: "starting" });
  send({ type: "start_scan", victim_id: vid, actor: "medic" });
  logEvent(`Scan requested for ${vid}`);
  renderScene();

  // Safety net: if neither scan_ready nor scan_error arrives (e.g. a
  // dropped websocket frame), release the button automatically so the
  // user isn't locked out.
  const prev = scanWatchdogs.get(vid);
  if (prev) clearTimeout(prev);
  scanWatchdogs.set(
    vid,
    setTimeout(() => {
      if (pendingScans.has(vid)) {
        pendingScans.delete(vid);
        scanProgress.delete(vid);
        logEvent(`Scan timed out for ${vid} — you can retry`);
        renderScene();
      }
      scanWatchdogs.delete(vid);
    }, SCAN_WATCHDOG_MS),
  );
}

function updateScanTileProgress(vid) {
  const prog = scanProgress.get(vid);
  if (!prog) return;
  const tile = document.querySelector(`.tile[data-vid="${cssEscape(vid)}"]`);
  if (!tile) return;
  const phaseLabel = {
    starting: "Starting…",
    sweeping: "Sweeping body",
    analyzing: "Analyzing",
    face: "Matching face",
    finalizing: "Finalizing",
  }[prog.phase] || "Scanning…";
  const pct = prog.total > 0 ? Math.min(100, Math.round((prog.step / prog.total) * 100)) : 0;
  const btn = tile.querySelector(".scan-btn");
  if (btn) btn.textContent = `${phaseLabel} · ${pct}%`;
  const lbl = tile.querySelector(".scan-sweep-label");
  if (lbl) lbl.textContent = phaseLabel;
}

function cssEscape(s) {
  return window.CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&");
}

function formatClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const mm = String(m % 60).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

// ---------- Detail panel ----------

function openDetail(vid) {
  selectedVictimId = vid;
  const v = scene.victims.find((x) => x.id === vid);
  selectedScanId = v?.last_scan_id || null;
  renderDetail();
  document.getElementById("victim-detail").classList.remove("hidden");
}

function closeDetail() {
  selectedVictimId = null;
  selectedScanId = null;
  disposeBodyViewer();
  document.getElementById("victim-detail").classList.add("hidden");
}

function buildPatientQrPayload(v, selectedScan) {
  const scan = selectedScan || null;
  const stableTag = scan?.salt_tag || (v.salt_tag_confirmed ? v.salt_tag : "UNCONFIRMED");
  const stablePriority = scan?.priority || (v.salt_tag_confirmed ? (v.priority || "P5") : "UNCONFIRMED");
  const stableWounds = scan?.wounds || (v.last_scan_id ? [] : []);
  return JSON.stringify({
    id: v.id,
    salt_tag: stableTag,
    priority: stablePriority,
    tbsa_burn_percent: scan?.tbsa_burn_percent ?? v.tbsa_burn_percent ?? null,
    scan_id: scan?.scan_id || v.last_scan_id || null,
    wounds: stableWounds.map((w) => ({
      label: w.label,
      body: w.body_region || w.body_location,
      severity: w.severity,
    })),
  });
}

function renderPatientQr(v, selectedScan = null) {
  const el = document.getElementById("patient-qr");
  if (!el) return;
  const qrLib = globalThis.qrcode;
  if (typeof qrLib !== "function") {
    el.textContent = "";
    return;
  }
  try {
    const payload = buildPatientQrPayload(v, selectedScan);
    if (el.dataset.payload === payload) return;
    const qr = qrLib(0, "M");
    qr.addData(payload);
    qr.make();
    el.dataset.payload = payload;
    el.innerHTML = qr.createImgTag(4);
  } catch {
    delete el.dataset.payload;
    el.textContent = "";
  }
}

function renderDetail() {
  const v = scene.victims.find((x) => x.id === selectedVictimId);
  const root = document.getElementById("modal-content");
  if (!v) {
    root.innerHTML = "";
    return;
  }

  const selectedScan = (v.scans || []).find((s) => s.scan_id === selectedScanId) ||
    (v.scans || [])[v.scans?.length - 1] || null;

  const march = (selectedScan?.march) || v.march || {};
  const marchBlock = ["M", "A", "R", "C", "H"]
    .map((k) => {
      const f = march[k] || { status: "unknown" };
      return `<div class="march-detail-row"><b>${MARCH_LABELS[k]}</b> — <code>${f.status}</code><br><span class="muted small">${escapeHtml(f.reason || "")}</span></div>`;
    })
    .join("");

  const mist = v.mist;
  const mistBlock = mist
    ? `<div class="mist-block">M  ${escapeHtml(mist.mechanism || "")}
I  ${(mist.injuries || []).map(escapeHtml).join(", ") || "—"}
S  HR=${mist.signs?.hr ?? "?"}  RR=${mist.signs?.rr ?? "?"}  SpO₂=${mist.signs?.spo2 ?? "?"}  ${CONSCIOUSNESS_LABELS[mist.signs?.consciousness] || "?"}
T  ${(mist.treatment || []).map(escapeHtml).join(", ") || "—"}
Notes: ${escapeHtml(mist.notes || "—")}
Source: ${escapeHtml(mist.source || "?")}</div>
${mist.dd1380 ? `<details class="dd1380"><summary>DD-1380 handoff JSON</summary><pre>${escapeHtml(JSON.stringify(mist.dd1380, null, 2))}</pre></details>` : ""}`
    : `<div class="muted" style="margin-top:8px">No MIST card yet. Tap "Generate MIST".</div>`;

  const bloodExtra = (v.blood_regions || [])
    .filter((b) => b.is_floor_pool)
    .map((b) => `${b.volume_bucket || "?"} pool`)
    .join(", ");

  const tbsa = selectedScan?.tbsa_burn_percent ?? v.tbsa_burn_percent;
  const tbsaLine =
    tbsa != null
      ? `<div class="tbsa-line">Estimated TBSA (burns): <b>${tbsa}%</b></div>`
      : "";

  const codewords = (v.tccc_codewords || []);
  const codewordsBlock = codewords.length
    ? `<div class="panel-title" style="margin-top:16px">TCCC codewords</div>
       <div class="codewords-row">${codewords.map((c) => `<span class="codeword-chip big">${escapeHtml(c)}</span>`).join("")}</div>`
    : "";

  const priorityRibbon = `<div class="priority-ribbon" style="background:${PRIORITY_COLORS[v.priority || "P5"]}">Priority ${v.priority || "P5"}</div>`;
  const tag = selectedScan?.salt_tag || v.salt_tag;
  const tagReason = selectedScan?.salt_reason || v.salt_tag_reason;

  root.innerHTML = `
    <div class="detail-header">
      ${faceAvatarHtml(v, 56)}
      <div>
        <h2 class="detail-title">${escapeHtml(v.id)}
          <span class="tag-chip tag-${tag}" style="margin-left:8px">${tag}${v.salt_tag_confirmed ? " [confirmed]" : " (suggested)"}</span>
        </h2>
        <div class="muted small">${escapeHtml(tagReason || "")}</div>
      </div>
    </div>
    ${priorityRibbon}
    ${tbsaLine}

    ${scanSummaryBlock(v, selectedScan)}

    <div class="panel-title" style="margin-top:16px">3D body & wounds</div>
    <label class="heatmap-label"><input type="checkbox" id="heatmap-toggle" /> Wound heatmap (decal)</label>
    <div id="body-3d-container" class="body-3d-container"></div>
    <div class="panel-title" style="margin-top:8px">2D graph (fallback)</div>
    ${avatarSvg(v, selectedScan)}

    <div class="panel-title" style="margin-top:16px">MARCH</div>
    ${marchBlock}

    <div class="panel-title" style="margin-top:16px">Wounds</div>
    ${
      (scanWounds(selectedScan) || v.wound_regions || [])
        .map(
          (w) =>
            `<span class="wound-badge">${escapeHtml(w.label || "?")} · ${escapeHtml(humanizeRegion(w.body_region || w.body_location) || "?")} · sev ${escapeHtml(w.severity || "?")} · ${((w.confidence || 0)).toFixed(2)}</span>`
        )
        .join(" ") || "<span class='muted'>—</span>"
    }
    ${bloodExtra ? `<div class="muted small" style="margin-top:6px">Floor blood: ${escapeHtml(bloodExtra)}</div>` : ""}

    ${codewordsBlock}

    <div class="panel-title" style="margin-top:16px">Patient tag (QR)</div>
    <div id="patient-qr" class="patient-qr"></div>

    <div class="panel-title" style="margin-top:16px">Vitals</div>
    <div>HR: <b>${selectedScan?.vitals?.hr ?? v.vitals?.hr ? Math.round(selectedScan?.vitals?.hr ?? v.vitals?.hr) + " bpm" : "pending"}</b>
    · RR: <b>${selectedScan?.vitals?.rr ?? v.vitals?.rr ? Math.round(selectedScan?.vitals?.rr ?? v.vitals?.rr) : "pending"}</b>
    · SpO₂: <b>pending</b></div>

    <div class="panel-title" style="margin-top:16px">Transcript</div>
    <div class="transcript" style="max-height:120px">${escapeHtml(selectedScan?.transcript_snippet || v.transcript || "—")}</div>

    <div class="panel-title" style="margin-top:16px">MIST card</div>
    ${mistBlock}

    <input type="text" class="note-input" id="note-text" placeholder="Add a note (e.g. TQ on 14:02)" />

    <div class="actions">
      <button type="button" class="btn primary" id="btn-scan">${v.scans?.length ? "Rescan victim" : "Scan this victim"}</button>
      <button type="button" class="btn" id="btn-mist">Generate MIST</button>
      <button type="button" class="btn" id="btn-handoff">Play handoff</button>
      <button type="button" class="btn" id="btn-tq">Start TQ timer (2h)</button>
      <button type="button" class="btn" id="btn-note">Save note</button>
    </div>

    <div class="panel-title" style="margin-top:16px">Confirm SALT tag</div>
    <div class="actions">
      <button type="button" class="btn red" data-tag="RED">Immediate</button>
      <button type="button" class="btn yellow" data-tag="YELLOW">Delayed</button>
      <button type="button" class="btn green" data-tag="GREEN">Minimal</button>
      <button type="button" class="btn grey" data-tag="GREY">Expectant</button>
      <button type="button" class="btn black" data-tag="BLACK">Dead</button>
    </div>
    <p class="hint">Grey and Black require explicit confirmation — AI will never auto-emit these.</p>
  `;

  const vid = v.id;
  document.getElementById("btn-scan")?.addEventListener("click", () => triggerScan(vid));
  document.getElementById("btn-mist")?.addEventListener("click", () => generateMist(vid));
  document.getElementById("btn-handoff")?.addEventListener("click", () => playHandoff(vid));
  document.getElementById("btn-tq")?.addEventListener("click", () => startTourniquetTimer(vid));
  document.getElementById("btn-note")?.addEventListener("click", () => submitNote(vid));
  root.querySelectorAll("[data-tag]").forEach((btn) => {
    btn.addEventListener("click", () => confirmTag(vid, btn.getAttribute("data-tag")));
  });
  root.querySelectorAll(".scan-timeline-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedScanId = btn.dataset.scanId;
      renderDetail();
    });
  });
  root.querySelectorAll(".legend-act").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const action = btn.getAttribute("data-action");
      const sid = btn.getAttribute("data-scan-id");
      const widx = btn.getAttribute("data-wound-idx");
      if (!sid || widx == null) return;
      const decision = action === "reject-wound" ? "rejected" : "confirmed";
      confirmWound(vid, sid, widx, decision);
    });
  });
  document.getElementById("heatmap-toggle")?.addEventListener("change", () => {
    disposeBodyViewer();
    const wounds = scanWounds(selectedScan) || v.wound_regions || [];
    syncBodyViewer(v, wounds, v.keypoints || []);
  });

  renderPatientQr(v, selectedScan);
  requestAnimationFrame(() => {
    const wounds = scanWounds(selectedScan) || v.wound_regions || [];
    syncBodyViewer(v, wounds, v.keypoints || []);
  });
}

function scanWounds(scan) {
  if (!scan) return null;
  return (scan.wounds || []).map((w) => ({
    label: w.label,
    body_location: w.body_region,
    severity: w.severity,
    confidence: w.confidence,
    bbox: w.bbox,
  }));
}

function scanSummaryBlock(v, scan) {
  if (!scan) {
    const hint = v.scans?.length
      ? ""
      : `<div class="scan-empty">
           <p>No scans captured yet for <b>${escapeHtml(v.id)}</b>.</p>
           <p class="muted small">In SCAN mode records are only saved when you press "Scan this victim".</p>
         </div>`;
    return hint;
  }

  const ts = new Date(scan.timestamp * 1000).toLocaleTimeString();
  const cropUrl = scan.crop_url;
  const frameUrl = scan.frame_url;
  const [fh, fw] = scan.frame_shape || [720, 1280];
  const [bx1, by1, bx2, by2] = scan.bbox || [0, 0, 0, 0];

  const arrows = (scan.wounds || []).map((w, i) => woundArrowSvg(w, scan.bbox, i)).join("");

  const timeline = (v.scans || [])
    .slice()
    .reverse()
    .map((s) => {
      const active = s.scan_id === scan.scan_id ? " active" : "";
      const t = new Date(s.timestamp * 1000).toLocaleTimeString();
      return `<button type="button" class="scan-timeline-item${active}" data-scan-id="${escapeHtml(s.scan_id)}">
        <b>${escapeHtml(s.priority || "P5")}</b> · ${t} · ${(s.wounds || []).length} wound${(s.wounds||[]).length===1?"":"s"}
      </button>`;
    })
    .join("");

  const frameAspect = (fh / fw) * 100;
  const keywordChips = (scan.keywords || [])
    .map((k) => {
      const lower = k.toLowerCase();
      const possible = lower.startsWith("possible ");
      const critical = /\b(critical|serious)\b/.test(lower);
      const cls = possible
        ? "codeword-chip possible"
        : critical
          ? "codeword-chip critical"
          : "codeword-chip";
      return `<span class="${cls}">${escapeHtml(k)}</span>`;
    })
    .join("");
  const sweepInfo = scan.sweep_frames && scan.sweep_frames > 1
    ? `<span class="muted small">Head-to-toe sweep · ${scan.sweep_frames} frames over ${(scan.sweep_duration_sec||0).toFixed(1)}s</span>`
    : `<span class="muted small">Single-shot capture</span>`;
  return `
    <div class="panel-title" style="margin-top:14px">Scan summary · ${escapeHtml(ts)}</div>
    <div class="scan-meta-row">${sweepInfo}</div>
    ${keywordChips ? `<div class="codewords-row" style="margin:6px 0 10px 0">${keywordChips}</div>` : ""}
    <div class="scan-summary">
      <div class="scan-crop" style="padding-top:${((by2-by1)/(bx2-bx1)*100).toFixed(2)}%">
        ${cropUrl ? `<img class="scan-crop-img" src="${cropUrl}" alt="victim crop"/>` : ""}
        <svg class="scan-arrow-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
          ${(scan.wounds || []).map((w, i) => cropArrowSvg(w, scan.bbox, i)).join("")}
        </svg>
      </div>
      <div class="scan-legend">
        ${(scan.wounds || []).map((w, i) => {
          const sev = (w.severity || "unknown").toLowerCase();
          const conf = typeof w.confidence === "number" ? w.confidence : 0;
          const pct = Math.round(conf * 100);
          const decision = (w.confirmation || "pending").toLowerCase();
          const isConfirmed = decision === "confirmed";
          const isRejected = decision === "rejected";
          const possible = (sev === "possible" || conf < 0.55) && !isConfirmed;
          const confClass = conf >= 0.75 ? "strong" : conf >= 0.55 ? "medium" : "weak";
          const sevLabel = possible ? "possible" : sev;
          const rowCls = [
            "scan-legend-row",
            possible ? "possible" : "",
            isConfirmed ? "confirmed" : "",
            isRejected ? "rejected" : "",
          ].filter(Boolean).join(" ");
          const btns = isRejected
            ? `<button type="button" class="legend-act undo" data-action="confirm-wound" data-scan-id="${escapeHtml(scan.scan_id)}" data-vid="${escapeHtml(v.id)}" data-wound-idx="${i}" title="Unreject">↺</button>`
            : `<button type="button" class="legend-act accept${isConfirmed ? " active" : ""}" data-action="confirm-wound" data-scan-id="${escapeHtml(scan.scan_id)}" data-vid="${escapeHtml(v.id)}" data-wound-idx="${i}" title="Confirm finding">✓</button>
               <button type="button" class="legend-act reject" data-action="reject-wound" data-scan-id="${escapeHtml(scan.scan_id)}" data-vid="${escapeHtml(v.id)}" data-wound-idx="${i}" title="Reject false positive">✗</button>`;
          return `<div class="${rowCls}">
            <span class="sev-dot sev-${escapeHtml(sevLabel)}" data-i="${i}"></span>
            <span class="legend-main">${escapeHtml(w.label)} · ${escapeHtml(humanizeRegion(w.body_region) || "?")}</span>
            <span class="legend-sev sev-${escapeHtml(sevLabel)}">${isConfirmed ? "CONFIRMED" : escapeHtml(sevLabel)}</span>
            <span class="legend-conf conf-${confClass}" title="GDINO confidence">${pct}%</span>
            <span class="legend-actions">${btns}</span>
          </div>`;
        }).join("") || "<span class='muted'>no wounds detected in this scan</span>"}
      </div>
      ${frameUrl ? `
      <div class="scan-frame" style="padding-top:${frameAspect.toFixed(2)}%">
        <img class="scan-frame-img" src="${frameUrl}" alt="scan frame"/>
        <svg class="scan-frame-svg" viewBox="0 0 ${fw} ${fh}" preserveAspectRatio="none">
          <rect x="${bx1}" y="${by1}" width="${bx2-bx1}" height="${by2-by1}" fill="none" stroke="#4da3ff" stroke-width="3"/>
          ${arrows}
        </svg>
      </div>` : ""}
    </div>
    ${timeline ? `<div class="panel-title" style="margin-top:10px">Scan timeline</div><div class="scan-timeline">${timeline}</div>` : ""}
  `;
}

function woundArrowSvg(w, bbox, idx) {
  const [bx1, by1, bx2, by2] = bbox || [0, 0, 0, 0];
  const [ax, ay] = w.arrow_anchor || [(bx1+bx2)/2, (by1+by2)/2];
  const [lx, ly] = w.label_anchor || [ax, ay];
  const color = severityColor(w.severity);
  return `
    <line x1="${lx}" y1="${ly}" x2="${ax}" y2="${ay}" stroke="${color}" stroke-width="2" stroke-linecap="round" marker-end="url(#arr-${idx})" />
    <defs>
      <marker id="arr-${idx}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${color}" />
      </marker>
    </defs>
    <circle cx="${ax}" cy="${ay}" r="6" fill="none" stroke="${color}" stroke-width="2"/>
    <text x="${lx}" y="${ly - 6}" fill="${color}" font-size="18" font-family="system-ui" text-anchor="${lx < (bx1+bx2)/2 ? 'start' : 'end'}">${escapeHtml(w.label)}</text>
  `;
}

function cropArrowSvg(w, bbox, idx) {
  // Map bbox coords into 0-100 viewBox of the crop.
  const [bx1, by1, bx2, by2] = bbox || [0, 0, 0, 0];
  const bw = Math.max(1, bx2 - bx1);
  const bh = Math.max(1, by2 - by1);
  const [ax, ay] = w.arrow_anchor || [(bx1+bx2)/2, (by1+by2)/2];
  const cx = ((ax - bx1) / bw) * 100;
  const cy = ((ay - by1) / bh) * 100;
  const color = severityColor(w.severity);
  // Anchor label toward nearer edge.
  const lx = cx < 50 ? 4 : 96;
  const ly = Math.max(8, Math.min(95, cy));
  return `
    <line x1="${lx}" y1="${ly}" x2="${cx}" y2="${cy}" stroke="${color}" stroke-width="0.6" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="1.6" fill="${color}"/>
    <text x="${lx}" y="${ly - 1.5}" fill="${color}" font-size="3" font-family="system-ui" text-anchor="${cx < 50 ? 'start' : 'end'}">${escapeHtml(w.label)} · ${escapeHtml(w.severity || '')}</text>
  `;
}

function severityColor(sev) {
  // Aligned with DESIGN.md triage palette (p1-p5 / severity tokens).
  switch ((sev || "").toLowerCase()) {
    case "critical": return "#ffb3b3";  // --p1-primary
    case "serious":  return "#ff9e5e";  // --sev-serious
    case "moderate": return "#fae500";  // --p2-fixed
    case "minor":    return "#00e475";  // --p3-tertiary
    case "possible": return "#c4c7cd";  // neutral, hollow look
    default:         return "#8a95a5";  // --on-surface-dim
  }
}

function avatarSvg(v, scan) {
  const wounds = scanWounds(scan) || v.wound_regions || [];
  const silhouette = silhouetteLayer();
  const pose = poseOverlayLayer(v.keypoints || []);
  const pins = woundPinsLayer(wounds);
  const hasPose = pose !== "";
  const badge = hasPose ? "WOUND TOPOLOGY MATRIX · POSE LOCKED" : "WOUND TOPOLOGY MATRIX";
  const priority = v.priority || "P5";

  return `<svg class="avatar-svg wireframe" viewBox="0 0 200 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Body wireframe graph">
    <defs>
      <linearGradient id="wf-bg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#0a0d11"/>
        <stop offset="1" stop-color="#060809"/>
      </linearGradient>
      <radialGradient id="wf-halo" cx="50%" cy="45%" r="55%">
        <stop offset="0"   stop-color="rgba(255,180,171,0.15)"/>
        <stop offset="0.7" stop-color="rgba(255,180,171,0.04)"/>
        <stop offset="1"   stop-color="rgba(0,0,0,0)"/>
      </radialGradient>
      <pattern id="wf-grid" width="16" height="16" patternUnits="userSpaceOnUse">
        <path d="M 16 0 L 0 0 L 0 16" fill="none" stroke="rgba(196,199,205,0.06)" stroke-width="0.5"/>
      </pattern>
      <filter id="wf-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <rect x="0" y="0" width="200" height="260" fill="url(#wf-bg)" rx="8"/>
    <rect x="0" y="0" width="200" height="260" fill="url(#wf-grid)" rx="8"/>
    ${priority === "P1" ? '<rect x="0" y="0" width="200" height="260" fill="url(#wf-halo)" rx="8"/>' : ""}
    <line x1="100" y1="18" x2="100" y2="248" stroke="rgba(196,199,205,0.08)" stroke-width="0.5" stroke-dasharray="2 4"/>
    <line x1="14" y1="132" x2="186" y2="132" stroke="rgba(196,199,205,0.08)" stroke-width="0.5" stroke-dasharray="2 4"/>
    ${silhouette}
    ${pose}
    ${pins}
    <g class="wf-corner">
      <path d="M6 6 L14 6 L14 8 L8 8 L8 14 L6 14 Z" fill="rgba(196,199,205,0.35)"/>
      <path d="M194 6 L186 6 L186 8 L192 8 L192 14 L194 14 Z" fill="rgba(196,199,205,0.35)"/>
      <path d="M6 254 L14 254 L14 252 L8 252 L8 246 L6 246 Z" fill="rgba(196,199,205,0.35)"/>
      <path d="M194 254 L186 254 L186 252 L192 252 L192 246 L194 246 Z" fill="rgba(196,199,205,0.35)"/>
    </g>
    <text x="100" y="252" text-anchor="middle" fill="rgba(196,199,205,0.45)" font-size="7" letter-spacing="1.2" font-family="Space Grotesk, monospace">${badge}</text>
  </svg>`;
}

function silhouetteLayer() {
  // Wireframe silhouette — hollow shapes, thin strokes, contour lines.
  const stroke = "rgba(196,199,205,0.42)";
  const innerStroke = "rgba(196,199,205,0.2)";
  return `
    <g class="wf-silhouette" fill="none" stroke-linejoin="round">
      <!-- Head + jaw + neck -->
      <circle cx="100" cy="36" r="16" stroke="${stroke}" stroke-width="1"/>
      <path d="M92 50 Q100 58 108 50" stroke="${stroke}" stroke-width="0.8"/>
      <path d="M96 54 L96 64 L104 64 L104 54" stroke="${stroke}" stroke-width="0.8"/>
      <!-- Shoulders + torso -->
      <path d="M72 68 L100 62 L128 68 L138 96 L138 152 L134 162 L66 162 L62 152 L62 96 Z"
            stroke="${stroke}" stroke-width="1"/>
      <!-- Torso contour lines (ribs / centerline hint) -->
      <path d="M72 92 L128 92" stroke="${innerStroke}" stroke-width="0.5"/>
      <path d="M72 108 L128 108" stroke="${innerStroke}" stroke-width="0.5"/>
      <path d="M72 128 L128 128" stroke="${innerStroke}" stroke-width="0.5"/>
      <path d="M100 68 L100 160" stroke="${innerStroke}" stroke-width="0.4" stroke-dasharray="1 2"/>
      <!-- Arms -->
      <path d="M72 68 L52 78 L42 132 L44 160 L52 162 L56 136 L64 92" stroke="${stroke}" stroke-width="1"/>
      <path d="M128 68 L148 78 L158 132 L156 160 L148 162 L144 136 L136 92" stroke="${stroke}" stroke-width="1"/>
      <!-- Hands -->
      <circle cx="48" cy="168" r="6" stroke="${stroke}" stroke-width="0.8"/>
      <circle cx="152" cy="168" r="6" stroke="${stroke}" stroke-width="0.8"/>
      <!-- Hips + legs -->
      <path d="M70 162 L76 240 L96 244 L100 180 L104 244 L124 240 L130 162" stroke="${stroke}" stroke-width="1"/>
      <!-- Knee markers -->
      <circle cx="86" cy="204" r="3" stroke="${innerStroke}" stroke-width="0.6"/>
      <circle cx="114" cy="204" r="3" stroke="${innerStroke}" stroke-width="0.6"/>
    </g>`;
}

function poseOverlayLayer(kp) {
  const visible = kp.filter((k) => k && k.length >= 3 && k[2] > 0.3);
  if (visible.length < 2) return "";

  const nose = kp[0];
  const lSh = kp[5], rSh = kp[6];
  const lHip = kp[11], rHip = kp[12];

  const ANCHORS = {
    nose: [100, 40], lSh: [70, 62], rSh: [130, 62], lHip: [85, 155], rHip: [115, 155],
  };
  const have = {
    nose: nose && nose[2] > 0.3, lSh: lSh && lSh[2] > 0.3, rSh: rSh && rSh[2] > 0.3,
    lHip: lHip && lHip[2] > 0.3, rHip: rHip && rHip[2] > 0.3,
  };
  const anchorPairs = [];
  if (have.nose) anchorPairs.push([nose, ANCHORS.nose]);
  if (have.lSh) anchorPairs.push([lSh, ANCHORS.lSh]);
  if (have.rSh) anchorPairs.push([rSh, ANCHORS.rSh]);
  if (have.lHip) anchorPairs.push([lHip, ANCHORS.lHip]);
  if (have.rHip) anchorPairs.push([rHip, ANCHORS.rHip]);
  if (anchorPairs.length < 2) return "";

  let sumSx = 0, sumSy = 0, sumAx = 0, sumAy = 0;
  for (const [src, dst] of anchorPairs) {
    sumSx += src[0]; sumSy += src[1]; sumAx += dst[0]; sumAy += dst[1];
  }
  const n = anchorPairs.length;
  const srcCx = sumSx / n, srcCy = sumSy / n;
  const dstCx = sumAx / n, dstCy = sumAy / n;
  let srcSpread = 0, dstSpread = 0;
  for (const [src, dst] of anchorPairs) {
    srcSpread += Math.hypot(src[0] - srcCx, src[1] - srcCy);
    dstSpread += Math.hypot(dst[0] - dstCx, dst[1] - dstCy);
  }
  const scale = dstSpread > 0 && srcSpread > 0 ? dstSpread / srcSpread : 1;
  const toX = (x) => dstCx + (x - srcCx) * scale;
  const toY = (y) => dstCy + (y - srcCy) * scale;

  const edges = [[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16]];
  const lines = edges.map(([a, b]) => {
    const ka = kp[a], kb = kp[b];
    if (!ka || !kb || ka[2] < 0.3 || kb[2] < 0.3) return "";
    return `<line x1="${toX(ka[0])}" y1="${toY(ka[1])}" x2="${toX(kb[0])}" y2="${toY(kb[1])}" stroke="rgba(255,180,171,0.7)" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.85" stroke-dasharray="2 2" />`;
  }).join("");
  const dots = kp.map((k) => k && k.length >= 3 && k[2] > 0.3
    ? `<circle cx="${toX(k[0])}" cy="${toY(k[1])}" r="1.8" fill="#ffb3b3" fill-opacity="0.95"/>` : "").join("");
  return `<g class="pose-overlay">${lines}${dots}</g>`;
}

function normRegionKey(loc) {
  return String(loc || "unknown").toLowerCase().replace(/\s+/g, "_");
}

function woundPinsLayer(wounds) {
  // Coordinates match the new 200×260 wireframe silhouette.
  const REGIONS = {
    head: [100, 32], face: [100, 38], neck: [100, 58],
    chest: [100, 98], upper_torso: [100, 92],
    abdomen: [100, 130], lower_torso: [100, 140], back: [100, 104],
    left_shoulder: [76, 72], right_shoulder: [124, 72],
    left_upper_arm: [58, 96], right_upper_arm: [142, 96],
    left_arm: [52, 108], right_arm: [148, 108],
    left_forearm: [48, 132], right_forearm: [152, 132],
    left_hand: [48, 168], right_hand: [152, 168],
    left_torso: [84, 112], right_torso: [116, 112],
    groin: [100, 166], pelvis: [100, 170],
    left_thigh: [88, 186], right_thigh: [112, 186],
    left_knee: [86, 204], right_knee: [114, 204],
    left_leg: [84, 220], right_leg: [116, 220],
    left_shin: [84, 222], right_shin: [116, 222],
    left_lower_leg: [84, 222], right_lower_leg: [116, 222],
    left_ankle: [82, 234], right_ankle: [118, 234],
    left_foot: [82, 242], right_foot: [118, 242],
    torso: [100, 120],
    unknown: [100, 130],
  };

  return wounds.map((wr, i) => {
    const loc = normRegionKey(wr.body_location || wr.body_region);
    const anchor = REGIONS[loc] || REGIONS.torso;
    const jitter = ((wr.label || "x").charCodeAt(0) % 5) - 2;
    const [px, py] = [anchor[0] + jitter, anchor[1] + jitter];
    const sev = (wr.severity || "unknown").toLowerCase();
    const color = severityColor(wr.severity);
    const isCritical = sev === "critical" || sev === "serious";

    // DESIGN.md §5: entry wound = primary circle + 4px surface inner stroke.
    // We use severity-colored ring + dark inner "surface" disk; crit adds
    // a crosshair + faint pulsing halo.
    const halo = isCritical
      ? `<circle cx="${px}" cy="${py}" r="12" fill="none" stroke="${color}" stroke-width="0.5" stroke-opacity="0.55"><animate attributeName="r" values="9;14;9" dur="1.8s" repeatCount="indefinite"/><animate attributeName="stroke-opacity" values="0.6;0;0.6" dur="1.8s" repeatCount="indefinite"/></circle>`
      : "";
    const crosshair = isCritical
      ? `<line x1="${px - 10}" y1="${py}" x2="${px - 6}" y2="${py}" stroke="${color}" stroke-width="1"/>
         <line x1="${px + 6}"  y1="${py}" x2="${px + 10}" y2="${py}" stroke="${color}" stroke-width="1"/>
         <line x1="${px}" y1="${py - 10}" x2="${px}" y2="${py - 6}"  stroke="${color}" stroke-width="1"/>
         <line x1="${px}" y1="${py + 6}"  x2="${px}" y2="${py + 10}" stroke="${color}" stroke-width="1"/>`
      : "";
    return `<g class="wf-pin" filter="url(#wf-glow)">
      ${halo}
      <circle cx="${px}" cy="${py}" r="5.5" fill="${color}" fill-opacity="0.95"/>
      <circle cx="${px}" cy="${py}" r="2.5" fill="#0a0d11"/>
      ${crosshair}
      <title>${escapeHtml(wr.label || "wound")} · ${escapeHtml(loc)}${sev !== "unknown" ? " · " + escapeHtml(sev) : ""}</title>
    </g>`;
  }).join("");
}

// ---------- Actions ----------

function confirmTag(vid, tag) {
  send({ type: "confirm_tag", victim_id: vid, tag, actor: "medic" });
  logEvent(`Tag ${tag} confirmed for ${vid}`);
}

function generateMist(vid) {
  send({ type: "generate_mist", victim_id: vid });
  logEvent(`Requested MIST for ${vid}…`);
}

function submitNote(vid) {
  const input = document.getElementById("note-text");
  const text = (input?.value || "").trim();
  if (!text) return;
  send({ type: "note", victim_id: vid, text });
  input.value = "";
  logEvent(`Note added to ${vid}`);
}

function startTourniquetTimer(vid) {
  send({ type: "start_timer", victim_id: vid, kind: "tourniquet", duration_seconds: 7200, note: "" });
  logEvent(`TQ timer started for ${vid}`);
}

function playHandoff(vid) {
  const v = scene.victims.find((x) => x.id === vid);
  if (!v) return;
  const m = v.mist;
  const hr = v.vitals?.hr ? `Heart rate ${Math.round(v.vitals.hr)}.` : "Heart rate pending.";
  const tag = v.salt_tag === "UNTAGGED" ? "untagged" : v.salt_tag.toLowerCase();
  const priority = v.priority || "P5";
  const injuries = m?.injuries?.length
    ? m.injuries.slice(0, 3).join(", ")
    : (v.wound_regions || [])
        .slice(0, 3)
        .map((w) => `${w.label} ${humanizeRegion(w.body_location)}`)
        .join(", ");
  const text = `${v.id}. ${priority}. ${tag}. ${injuries || "no visible injuries"}. ${hr}`;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  speechSynthesis.cancel();
  speechSynthesis.speak(utter);
}

// ---------- Boot ----------

document.getElementById("scenario-select").addEventListener("change", (e) => {
  send({ type: "set_scenario", scenario: e.target.value, actor: "medic" });
  logEvent(`Scenario → ${e.target.value}`);
});

document.getElementById("profile-select").addEventListener("change", (e) => {
  const profile = e.target.value;
  send({ type: "set_profile", profile, actor: "medic" });
  logEvent(`Profile → ${profile} (hot-swap)`);
});

document.getElementById("scan-session-btn")?.addEventListener("click", () => {
  if (scanSession.active) {
    send({ type: "stop_auto_scan", actor: "medic" });
    logEvent("Auto scan session stopped");
  } else {
    send({ type: "start_auto_scan", actor: "medic" });
    logEvent("Auto scan session started");
  }
});

document.getElementById("mode-pill").addEventListener("click", (e) => {
  const btn = e.target.closest(".pill-opt");
  if (!btn) return;
  const mode = btn.dataset.mode;
  if (mode === operatingMode) return;
  if (mode === "live") {
    showConfirmModal("confirm-modal");
    return;
  }
  setOperatingMode(mode, true);
});

document.getElementById("confirm-live-cancel")?.addEventListener("click", () => {
  hideConfirmModal("confirm-modal");
});
document.getElementById("confirm-live-ok")?.addEventListener("click", () => {
  hideConfirmModal("confirm-modal");
  setOperatingMode("live", true);
});

document.getElementById("modal-close").addEventListener("click", closeDetail);
document.getElementById("detail-backdrop")?.addEventListener("click", closeDetail);

// --- Docked nav tabs (Scene / Matrix / Patients) ---
document.querySelectorAll(".nav-dock .tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-dock .tab").forEach((b) => b.removeAttribute("aria-current"));
    btn.setAttribute("aria-current", "page");
    const view = btn.dataset.view;
    document.body.dataset.view = view;
    const side = document.querySelector("aside.side");
    const mainPanel = document.querySelector("main.grid > section.panel");
    if (view === "matrix") {
      if (side) side.style.display = "none";
      if (mainPanel) mainPanel.style.gridColumn = "1 / -1";
      document.querySelector("main.grid").style.gridTemplateColumns = "1fr";
    } else if (view === "patients") {
      if (side) side.style.display = "none";
      document.querySelector("main.grid").style.gridTemplateColumns = "1fr";
      const firstP1 = scene.victims.find((x) => (x.priority || "P5") === "P1") || scene.victims[0];
      if (firstP1) openDetail(firstP1.id);
    } else {
      if (side) side.style.display = "";
      document.querySelector("main.grid").style.gridTemplateColumns = "";
    }
  });
});

const nightCb = document.getElementById("night-mode");
if (nightCb) {
  nightCb.checked = localStorage.getItem("mascal-night") === "1";
  document.body.classList.toggle("theme-night", nightCb.checked);
  nightCb.addEventListener("change", () => {
    localStorage.setItem("mascal-night", nightCb.checked ? "1" : "0");
    document.body.classList.toggle("theme-night", nightCb.checked);
  });
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

connect();
renderScanSession();
