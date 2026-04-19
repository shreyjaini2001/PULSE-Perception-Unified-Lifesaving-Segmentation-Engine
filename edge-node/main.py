"""MASCAL Triage Relay — edge node entry point.

Runs the full perception + clinical loop and serves the receiver dashboard
and ATAK bridge.

Usage:
    python edge-node/main.py
    python edge-node/main.py --source 0
    python edge-node/main.py --source edge-node/demo/scenario_blast.mp4
    python edge-node/main.py --scenario fire_structure --profile max
    python edge-node/main.py --mode live                # continuous update
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import signal
import sys
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\..*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

from audit import AuditLog  # noqa: E402
from broadcast import AtakBridge, BroadcastServer  # noqa: E402
from broadcast.atak_bridge import AtakConfig  # noqa: E402
from pipeline import (  # noqa: E402
    AudioTranscriber,
    BodyLocator,
    FaceReID,
    GdinoSam2Backend,
    MistGenerator,
    PersonDetector,
    RppgEstimator,
    Sam3Backend,
    ScanEngine,
    WoundSegmenter,
)
from state import (  # noqa: E402
    MarchState,
    Scene,
    SaltTag,
    Vitals,
    Victim,
    derive_march,
    derive_priority,
    estimate_tbsa_percent,
    scan_transcript,
    suggest_salt,
)


SCRIPT_DIR = HERE
CONFIG_DIR = SCRIPT_DIR / "config"
DASHBOARD_DIR = (SCRIPT_DIR.parent / "dashboard").resolve()


def load_config() -> Dict[str, Any]:
    with open(CONFIG_DIR / "runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenarios() -> Dict[str, Any]:
    with open(CONFIG_DIR / "scenarios.json", "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MASCAL Triage Relay — edge node")
    p.add_argument("--source", default=None,
                   help="Video source. 0 = default webcam, path = video file, rtsp:// = network")
    p.add_argument("--scenario", default=None,
                   help="Scenario mode id (see edge-node/config/scenarios.json)")
    p.add_argument("--mode", choices=("scan", "live"), default=None,
                   help="Operating mode. 'scan' (default) freezes per-victim records on demand; "
                        "'live' updates every frame.")
    p.add_argument("--profile", choices=("fast", "balanced", "max"), default=None,
                   help="Model-tier profile. Overrides runtime.yaml.")
    p.add_argument("--no-llm", action="store_true", help="Disable Llama MIST synthesis")
    p.add_argument("--no-sam", action="store_true", help="Disable Grounding DINO + SAM")
    p.add_argument("--no-audio", action="store_true", help="Disable microphone transcription")
    p.add_argument("--no-face", action="store_true", help="Disable InsightFace re-ID")
    p.add_argument("--no-atak", action="store_true", help="Disable ATAK CoT publisher")
    p.add_argument("--demo-wounds", action="store_true",
                   help="Loosen wound-evidence gates for monitor/video hackathon demos")
    p.add_argument("--headless", action="store_true", help="Skip cv2 preview window")
    p.add_argument("--fps", type=float, default=None, help="Override target FPS")
    return p.parse_args()


def _coerce_source(raw: Any) -> Any:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return raw


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _apply_profile(cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    """Overlay the selected profile onto the pipeline block (non-destructive)."""
    pcfg = copy.deepcopy(cfg.get("pipeline", {}))
    prof = (cfg.get("profiles") or {}).get(profile_name) or {}
    for k, v in prof.items():
        pcfg[k] = v
    cfg["pipeline"] = pcfg
    cfg["profile"] = profile_name
    return cfg


class EdgeNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config()
        self.scenarios = load_scenarios()

        if args.source is not None:
            self.cfg["capture"]["source"] = args.source
        if args.scenario:
            self.cfg["scenario"]["default"] = args.scenario
        if args.mode:
            self.cfg["operating_mode"] = args.mode
        # Profile overlay (CLI wins).
        profile_name = args.profile or self.cfg.get("profile", "balanced")
        self.cfg = _apply_profile(self.cfg, profile_name)

        if args.no_llm:
            self.cfg["pipeline"]["llm_enabled"] = False
        if args.no_sam:
            self.cfg["pipeline"]["use_grounding_dino"] = False
            self.cfg["pipeline"]["use_sam"] = False
        if args.no_audio:
            self.cfg["pipeline"]["audio_enabled"] = False
        if args.no_face:
            self.cfg["pipeline"]["face_reid_enabled"] = False
        if args.no_atak:
            self.cfg["atak"]["enabled"] = False
        if args.demo_wounds:
            self.cfg["pipeline"]["wound_demo_mode"] = True
        if args.fps:
            self.cfg["capture"]["target_fps"] = args.fps

        self.headless = args.headless
        # Some OpenCV wheels (opencv-python-headless, which InsightFace pulls
        # in on Windows) have highgui compiled out. We probe imshow once in
        # run() and auto-disable the preview instead of crashing the loop.
        self._preview_disabled = False
        self.mode: str = str(self.cfg.get("operating_mode", "scan"))

        scenario_id = self.cfg["scenario"]["default"]
        if scenario_id not in self.scenarios:
            print(f"[main] unknown scenario '{scenario_id}', falling back to combat_blast")
            scenario_id = "combat_blast"
        self.scenario_id = scenario_id

        try:
            import torch  # noqa: WPS433

            print(f"[main] torch CUDA available: {torch.cuda.is_available()}", flush=True)
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                print(
                    f"[main] GPU: {torch.cuda.get_device_name(0)}  "
                    f"VRAM: {props.total_memory / 1e9:.1f} GB",
                    flush=True,
                )
        except Exception as exc:
            print(f"[main] GPU diagnostics skipped: {exc}", flush=True)

        print(f"[main] profile={profile_name}  mode={self.mode}", flush=True)

        self.audit = AuditLog(self.cfg["logging"]["audit_dir"])
        self.scene = Scene(scenario=scenario_id)
        self._pipeline_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._shutdown = threading.Event()
        self._scan_run_lock = threading.Lock()
        self._last_wound_scan: Dict[str, float] = {}
        self._preview_wound_overlay: Dict[str, Tuple[Any, Any]] = {}
        self._last_face_reid: Dict[str, float] = {}
        self._track_to_victim: Dict[str, str] = {}
        self._auto_scan_active = False
        self._auto_scan_inflight = False
        self._auto_scan_target_id: Optional[str] = None
        self._auto_scan_last_completed: Dict[str, float] = {}

        # ---- Pipeline stages ----
        pcfg = self.cfg["pipeline"]
        print(f"[main] scenario={scenario_id}")
        self.person = None
        self.wounds = None
        self.body = None
        self.face = None
        self.rppg = None
        self.audio = None
        self.llm = None
        self.anomaly_prior = None
        self.scan_engine = None
        self._install_stage_bundle(self._build_stage_bundle(pcfg, scenario_id))

        # ---- Broadcast ----
        bcfg = self.cfg["broadcast"]
        self.broadcast = BroadcastServer(
            http_host=bcfg.get("http_host", "0.0.0.0"),
            http_port=bcfg.get("http_port", 8080),
            ws_host=bcfg.get("ws_host", "0.0.0.0"),
            ws_port=bcfg.get("ws_port", 8081),
            dashboard_dir=str(DASHBOARD_DIR),
            on_control=self._on_control,
        )

        acfg = self.cfg.get("atak", {})
        self.atak = AtakBridge(AtakConfig(
            enabled=acfg.get("enabled", False),
            host=acfg.get("host", "127.0.0.1"),
            port=acfg.get("port", 4242),
            anchor_lat=acfg.get("anchor_lat", 38.8895),
            anchor_lon=acfg.get("anchor_lon", -77.0353),
        ))

        self._frame_idx = 0
        self._wound_scan_interval = float(pcfg.get("wound_scan_interval_seconds", 1.0))
        # SCAN mode: optional OpenCV HUD wound preview (does not mutate victim
        # state or the dashboard — medic scan still owns persisted wounds).
        self._scan_wound_preview_enabled = bool(pcfg.get("scan_wound_preview_enabled", True))
        self._scan_wound_preview_interval = float(
            pcfg.get("scan_wound_preview_interval_seconds", 3.0))
        self._last_wound_preview_cycle_ts = 0.0
        self._preview_wound_victim_idx = 0
        self._face_reid_interval = float(pcfg.get("face_reid_interval_seconds", 0.5))
        self._last_broadcast = 0.0
        self._last_heartbeat = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_pts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_wound_segmenter(self, pcfg: Dict[str, Any], scenario_id: str) -> WoundSegmenter:
        """Build a segmenter wrapping the profile-selected backend."""
        gdino_backend = GdinoSam2Backend(
            gdino_model=pcfg.get("gdino_model", "IDEA-Research/grounding-dino-base"),
            sam_model=pcfg.get("sam_model", "facebook/sam2-hiera-small"),
            use_grounding_dino=pcfg.get("use_grounding_dino", True),
            use_sam=pcfg.get("use_sam", True),
            box_threshold=pcfg.get("gdino_box_threshold", 0.3),
            text_threshold=pcfg.get("gdino_text_threshold", 0.25),
        )
        backend: Any = gdino_backend
        if pcfg.get("wound_backend") == "sam3":
            backend = Sam3Backend(
                version=pcfg.get("sam3_version", "sam3.1"),
                delegate=gdino_backend,
                hf_checkpoint=pcfg.get("sam3_hf_checkpoint"),
            )
        return WoundSegmenter(
            gdino_prompt=self.scenarios[scenario_id]["gdino_prompts"],
            body_dilation_px=int(pcfg.get("body_dilation_px", 18)),
            min_body_overlap=float(pcfg.get("min_body_overlap", 0.4)),
            debug_enabled=bool(pcfg.get("wound_debug_enabled", False)),
            debug_dir=str(pcfg.get("wound_debug_dir", "logs/wound_debug")),
            demo_mode=bool(pcfg.get("wound_demo_mode", False)),
            backend=backend,
        )

    # ------------------------------------------------------------------
    def _build_stage_bundle(self, pcfg: Dict[str, Any], scenario_id: str) -> Dict[str, Any]:
        person = PersonDetector(
            model_name=f"{pcfg.get('person_detector', 'yolov8s-pose')}.pt"
            if not pcfg.get("person_detector", "").endswith(".pt") else pcfg["person_detector"],
            confidence=pcfg.get("person_confidence", 0.5),
            iou_threshold=pcfg.get("track_iou_threshold", 0.3),
            max_age_frames=pcfg.get("track_max_age_frames", 30),
            min_area_fraction=pcfg.get("person_min_area_fraction", 0.02),
            min_visible_keypoints=pcfg.get("person_min_visible_keypoints", 5),
            keypoint_conf_threshold=pcfg.get("person_keypoint_conf_threshold", 0.3),
            min_bbox_short_edge_px=int(pcfg.get("person_min_bbox_short_edge_px", 56)),
            max_aspect_width_over_height=float(pcfg.get("person_max_aspect_wh", 1.12)),
            min_keypoint_span_w=float(pcfg.get("person_min_kp_span_w", 0.22)),
            min_keypoint_span_h=float(pcfg.get("person_min_kp_span_h", 0.28)),
            min_shoulder_separation_frac_diag=float(
                pcfg.get("person_min_shoulder_sep_diag", 0.12)),
            suppress_contained_iou=float(pcfg.get("person_suppress_contained_iou", 0.88)),
            reject_ceiling_shorties=bool(pcfg.get("person_reject_ceiling_shorties", True)),
            ceiling_top_frac=float(pcfg.get("person_ceiling_top_frac", 0.10)),
            ceiling_max_height_frac=float(pcfg.get("person_ceiling_max_height_frac", 0.11)),
            bg_suppress_enabled=bool(pcfg.get("person_bg_suppress_enabled", True)),
            bg_min_population=int(pcfg.get("person_bg_min_population", 2)),
            bg_area_ratio_vs_max=float(pcfg.get("person_bg_area_ratio_vs_max", 0.32)),
            bg_max_confidence=float(pcfg.get("person_bg_max_confidence", 0.71)),
            bg_isolated_iou_cap=float(pcfg.get("person_bg_isolated_iou_cap", 0.03)),
            bg_isolated_area_ratio=float(pcfg.get("person_bg_isolated_area_ratio", 0.45)),
            bg_isolated_max_confidence=float(pcfg.get("person_bg_isolated_max_confidence", 0.68)),
        )
        wounds = self._build_wound_segmenter(pcfg, scenario_id)
        body = BodyLocator()
        face = FaceReID(
            name=pcfg.get("face_reid_model", "buffalo_l"),
            det_size=(pcfg.get("face_reid_det_size", 640),
                      pcfg.get("face_reid_det_size", 640)),
            threshold=pcfg.get("face_reid_threshold", 0.45),
        ) if pcfg.get("face_reid_enabled", True) else None
        rppg = RppgEstimator(
            window_seconds=pcfg.get("rppg_window_seconds", 10),
            fps_hint=self.cfg["capture"].get("target_fps", 10),
            min_confidence=pcfg.get("rppg_min_confidence", 0.4),
        ) if pcfg.get("rppg_enabled", True) else None
        audio = AudioTranscriber(
            model=pcfg.get("whisper_model", "tiny.en"),
            device=pcfg.get("whisper_device", "auto"),
            compute_type=pcfg.get("whisper_compute_type", "auto"),
        ) if pcfg.get("audio_enabled", True) else None
        llm = MistGenerator(
            model_path=pcfg.get("llm_path"),
            enabled=pcfg.get("llm_enabled", False),
            n_ctx=pcfg.get("llm_n_ctx", 2048),
            max_tokens=pcfg.get("llm_max_tokens", 256),
        )

        anomaly_prior = None
        if pcfg.get("anomaly_prior_enabled"):
            try:
                from pipeline.anomaly import DinoV3AnomalyPrior
                fb = pcfg.get("anomaly_model_fallbacks") or [
                    "facebook/dinov2-base",
                ]
                if isinstance(fb, str):
                    fb = [fb]
                anomaly_prior = DinoV3AnomalyPrior(
                    checkpoint=pcfg.get(
                        "anomaly_model",
                        "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    ),
                    fallbacks=fb,
                )
            except Exception as exc:
                print(f"[main] anomaly prior disabled ({exc})")
                anomaly_prior = None

        scan_engine = ScanEngine(
            wound_segmenter=wounds,
            body_locator=body,
            rppg=rppg,
            face_reid=face,
            llm=llm,
            scenarios=self.scenarios,
            burn_estimator=estimate_tbsa_percent,
            anomaly_prior=anomaly_prior,
            consensus_enabled=bool(pcfg.get("scan_consensus_enabled", False)),
            ensemble_enabled=bool(pcfg.get("ensemble_enabled", False)),
        )
        return {
            "person": person,
            "wounds": wounds,
            "body": body,
            "face": face,
            "rppg": rppg,
            "audio": audio,
            "llm": llm,
            "anomaly_prior": anomaly_prior,
            "scan_engine": scan_engine,
        }

    # ------------------------------------------------------------------
    def _seed_face_reid(self, face: Optional[FaceReID]) -> None:
        if face is None:
            return
        for victim in self.scene.all_victims():
            if victim.face_embedding:
                try:
                    face.seed_known(victim.id, victim.face_embedding)
                except Exception:
                    continue

    # ------------------------------------------------------------------
    def _install_stage_bundle(self, bundle: Dict[str, Any], restart_audio: Optional[bool] = None) -> None:
        old_audio = getattr(self, "audio", None)
        if restart_audio is None:
            restart_audio = bool(getattr(old_audio, "_running", False))

        self.person = bundle["person"]
        self.wounds = bundle["wounds"]
        self.body = bundle["body"]
        self.face = bundle["face"]
        self.rppg = bundle["rppg"]
        self.audio = bundle["audio"]
        self.llm = bundle["llm"]
        self.anomaly_prior = bundle["anomaly_prior"]
        self.scan_engine = bundle["scan_engine"]
        self._seed_face_reid(self.face)

        self._track_to_victim.clear()
        self._last_face_reid.clear()
        self._last_wound_scan.clear()
        self._preview_wound_overlay.clear()

        if old_audio is not None and old_audio is not self.audio:
            try:
                old_audio.stop()
            except Exception:
                pass
        if restart_audio and self.audio is not None:
            self.audio.start()

    # ------------------------------------------------------------------
    def _reload_profile(self, new_profile: str, actor: str = "medic") -> None:
        new_cfg = _apply_profile(copy.deepcopy(self.cfg), new_profile)
        new_cfg["profile"] = new_profile
        new_pcfg = new_cfg["pipeline"]
        restart_audio = bool(getattr(self.audio, "_running", False))

        try:
            bundle = self._build_stage_bundle(new_pcfg, self.scene.scenario)
        except Exception as exc:
            print(f"[main] profile reload failed ({exc})", flush=True)
            self.broadcast.broadcast({
                "type": "audit",
                "event": {
                    "kind": "profile_reload_failed",
                    "profile": new_profile,
                    "actor": actor,
                    "error": str(exc),
                },
            })
            return

        with self._pipeline_lock:
            prev = self.cfg.get("profile", "balanced")
            self.cfg = new_cfg
            self._wound_scan_interval = float(new_pcfg.get("wound_scan_interval_seconds", 1.0))
            self._scan_wound_preview_enabled = bool(
                new_pcfg.get("scan_wound_preview_enabled", True))
            self._scan_wound_preview_interval = float(
                new_pcfg.get("scan_wound_preview_interval_seconds", 3.0))
            self._face_reid_interval = float(new_pcfg.get("face_reid_interval_seconds", 0.5))
            self._install_stage_bundle(bundle, restart_audio=restart_audio)

        self.audit.write("profile_changed", actor=actor,
                         previous_state=prev, new_state=new_profile)
        self.broadcast.broadcast({"type": "profile", "profile": new_profile})
        print(f"[main] profile fully reloaded ({new_profile})", flush=True)

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.broadcast.start()
        if self.audio is not None:
            self.audio.start()

        import cv2

        source = _coerce_source(self.cfg["capture"]["source"])
        print(f"[main] opening source: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[main] ERROR: cannot open source {source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["capture"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["capture"]["height"])

        target_fps = self.cfg["capture"].get("target_fps", 10)
        frame_interval = 1.0 / target_fps

        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())

        last_tick = 0.0
        try:
            while not self._shutdown.is_set():
                ok, frame = cap.read()
                if not ok:
                    if isinstance(source, str) and not source.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.02)
                    continue

                if self.cfg["capture"].get("mirror"):
                    frame = cv2.flip(frame, 1)

                now = time.time()
                if now - last_tick < frame_interval:
                    continue
                last_tick = now

                self._process_frame(frame)

                if not self.headless and not self._preview_disabled:
                    self._render_preview(frame)
                    if not self._preview_disabled:
                        try:
                            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                                self._shutdown.set()
                        except cv2.error:
                            self._preview_disabled = True
        finally:
            cap.release()
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if self.audio is not None:
                self.audio.stop()
            self.broadcast.stop()
            self.audit.close()

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> None:
        import cv2

        self._frame_idx += 1
        self.scene.frame_count = self._frame_idx
        self.scene.last_frame_ts = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0
        if self._prev_gray is not None:
            if self._flow_pts is None or self._frame_idx % 24 == 0:
                self._flow_pts = cv2.goodFeaturesToTrack(
                    self._prev_gray, maxCorners=60, qualityLevel=0.02, minDistance=10,
                )
            if self._flow_pts is not None and len(self._flow_pts) >= 4:
                nxt, st, _e = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._flow_pts, None
                )
                if nxt is not None and st is not None:
                    sel = st.flatten() == 1
                    if sel.sum() > 3:
                        src_xy = self._flow_pts.reshape(-1, 2)
                        dst_xy = nxt.reshape(-1, 2)
                        delta = dst_xy[sel] - src_xy[sel]
                        dx = float(np.median(delta[:, 0]))
                        dy = float(np.median(delta[:, 1]))
        self._prev_gray = gray
        self.scene.camera_flow = {"dx": dx, "dy": dy}

        now = time.time()
        with self._pipeline_lock:
            tracked = self.person.process(frame)

            transcript = self.audio.recent_text(30.0) if self.audio else ""
            self.scene.global_transcript = transcript

            rppg_inputs: List[Tuple[str, Tuple[int, int, int, int]]] = []
            for t in tracked:
                x1, y1, x2, y2 = t.bbox
                face_h = int((y2 - y1) * 0.35)
                face_box = (x1, y1, x2, y1 + max(40, face_h))
                # Use track id; we remap to victim id after face re-ID below.
                rppg_inputs.append((t.track_id, face_box))

            rppg_results = self.rppg.process(frame, rppg_inputs) if self.rppg else {}

            visible_ids: set = set()
            for t in tracked:
                victim_id = self._resolve_victim_id(t, frame, now)
                visible_ids.add(victim_id)

                # Archive-aware get: if the victim existed (even off-screen), we
                # reuse that record — preserving its scans, timers, SALT tag,
                # etc. This is what makes a re-entry feel like a re-scan rather
                # than a new victim.
                victim = self.scene.get(victim_id)
                if victim is None:
                    victim = Victim(id=victim_id)
                else:
                    # Re-entry: welcome the victim back on screen.
                    if victim.off_screen and victim.scans:
                        print(f"[main] victim {victim_id} re-entered "
                              f"(scans={len(victim.scans)})", flush=True)
                        try:
                            self.broadcast.broadcast({
                                "type": "victim_recognized",
                                "victim_id": victim_id,
                                "scan_count": len(victim.scans),
                                "ts": now,
                            })
                        except Exception:
                            pass
                victim.bbox = t.bbox
                victim.keypoints = t.keypoints
                victim.touch()

                if self.mode == "live":
                    self._update_victim_live(victim, frame, t, now, transcript, rppg_results)
                else:
                    # SCAN mode: only refresh bbox + face thumb + passive telemetry.
                    # Do NOT update wound_regions / march / salt until the medic
                    # presses Scan. Vitals/ transcripts still populate so the HUD
                    # can display living signal without persisting decisions.
                    est = rppg_results.get(t.track_id)
                    if est:
                        victim.vitals.hr = est.hr
                        victim.vitals.rr = est.rr
                        victim.vitals.hr_confidence = est.hr_confidence
                        victim.vitals.rr_confidence = est.rr_confidence
                        victim.vitals.last_updated = est.timestamp or now
                    victim.transcript = transcript
                    victim.transcript_updated = now
                    codewords = [c["codeword"] for c in scan_transcript(transcript)]
                    victim.tccc_codewords = codewords
                    # Auto-start TQ timer on the "TQ applied" codeword — the
                    # 2hr ischemia clock starts the instant a medic says it out
                    # loud, not when they remember to press Start Timer later.
                    if any("TQ applied" == c for c in codewords):
                        self._maybe_auto_start_tq(victim, source="codeword",
                                                  note="transcript: TQ applied")

                if victim.geo_lat is None:
                    victim.geo_lat, victim.geo_lon = self._synth_geo(t.bbox, frame.shape[:2])

                # Priority always derivable (defaults to P5 if no data yet).
                victim.priority = derive_priority(victim.salt_tag.value, victim.wound_regions, victim.march)

                self.scene.upsert_victim(victim)

            # Throttled wound/blood pass for the local preview window only. In
            # SCAN mode ``victim.wound_regions`` stays empty until the medic
            # taps Scan — without this HUD pass the OpenCV overlay looked
            # "dead" even though GDINO+SAM were healthy.
            if self.mode == "scan" and self._scan_wound_preview_enabled and tracked:
                if now - self._last_wound_preview_cycle_ts >= self._scan_wound_preview_interval:
                    self._last_wound_preview_cycle_ts = now
                    idx = self._preview_wound_victim_idx % len(tracked)
                    self._preview_wound_victim_idx += 1
                    t_prev = tracked[idx]
                    pv_id = self._resolve_victim_id(t_prev, frame, now)
                    try:
                        w_prev, b_prev = self.wounds.process(frame, t_prev.bbox)
                        self._preview_wound_overlay[pv_id] = (w_prev, b_prev)
                    except Exception as exc:
                        if self._frame_idx % 120 == 0:
                            print(f"[main] HUD wound preview failed ({exc})", flush=True)

        # Archive model: anyone not tracked this frame is marked off-screen;
        # prune_stale then removes only victims with no scans. Scanned
        # victims stick around so a re-entry matches back to them.
        self.scene.mark_off_screen(visible_ids, now=now)
        self.scene.prune_stale(max_age_seconds=30.0)
        self._maybe_start_auto_scan(now)

        self._check_timer_milestones(now)

        if now - self._last_broadcast > 0.2:
            self._last_broadcast = now
            self.broadcast.broadcast({
                "type": "snapshot",
                "scene": self.scene.snapshot(),
                "mode": self.mode,
                "profile": self.cfg.get("profile", "balanced"),
                "scan_session": self._scan_session_state(),
            })
            if transcript:
                self.broadcast.broadcast({"type": "transcript", "text": transcript, "ts": now})

        if now - self._last_heartbeat > 5.0:
            self._last_heartbeat = now
            print(f"[main] frame={self._frame_idx} victims={len(self.scene.victims)} "
                  f"tracked_this_frame={len(tracked)} mode={self.mode}", flush=True)

        with self._frame_lock:
            self._latest_frame = frame

    # ------------------------------------------------------------------
    def _resolve_victim_id(self, tracked_person, frame: np.ndarray, now: float) -> str:
        """Map the tracker id to a face-anchored canonical victim id.

        Falls back to the tracker id when InsightFace is unavailable or no
        face is visible yet for this bbox.
        """
        tid = tracked_person.track_id
        cached = self._track_to_victim.get(tid)

        # Fast path: tracker id is still mapped to an *active* victim and we
        # re-verified the face recently — just return the cached id.
        if cached and cached in self.scene.victims:
            if not self.scene.victims[cached].off_screen:
                last = self._last_face_reid.get(tid, 0.0)
                if self.face is None or now - last < self._face_reid_interval:
                    return cached
        bbox_owner = self._find_recent_bbox_owner(
            tracked_person.bbox,
            now,
            exclude_id=cached,
        )

        # Anything else (new tracker id, cached victim went off-screen,
        # cached victim was pruned) → *always* ask InsightFace. This is what
        # reattaches an archived scanned victim to their face on re-entry.
        vid = cached or bbox_owner or tid
        if self.face is None or not self.face.ready:
            self._track_to_victim[tid] = vid
            return vid

        try:
            match = self.face.match_or_register(frame, tracked_person.bbox, fallback_id=vid)
        except Exception as exc:
            print(f"[main] face match failed ({exc})")
            match = None

        self._last_face_reid[tid] = now
        if match is None:
            self._track_to_victim[tid] = vid
            return vid

        new_vid = match.victim_id
        if cached and cached != new_vid and cached in self.scene.victims:
            # Tracker re-association — migrate the record ONLY if the new
            # canonical id is currently unused. Otherwise, keep the archived
            # victim intact (this branch used to orphan the scan history).
            stale = self.scene.victims.get(cached)
            if stale is not None and new_vid not in self.scene.victims:
                self.scene.victims.pop(cached, None)
                stale.id = new_vid
                self.scene.victims[new_vid] = stale
            elif stale is not None and new_vid in self.scene.victims:
                # The face says this is an existing archived victim. Drop
                # the duplicate placeholder that was created under the
                # tracker id.
                if not stale.scans:
                    self.scene.victims.pop(cached, None)
        self._track_to_victim[tid] = new_vid

        # Persist face embedding on the victim so re-launches can seed from audit.
        victim = self.scene.get(new_vid)
        if victim is not None and match.embedding is not None:
            victim.face_embedding = [float(x) for x in match.embedding.tolist()]
        return new_vid

    # ------------------------------------------------------------------
    def _find_recent_bbox_owner(
        self,
        bbox: Tuple[int, int, int, int],
        now: float,
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        """Reuse a recent live victim record when tracker IDs churn."""
        best_id: Optional[str] = None
        best_iou = 0.0
        for victim in self.scene.all_victims():
            if victim.id == exclude_id:
                continue
            if victim.off_screen:
                continue
            if now - float(victim.last_seen or 0.0) > 2.5:
                continue
            if victim.bbox == (0, 0, 0, 0):
                continue
            overlap = _bbox_iou(bbox, victim.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_id = victim.id
        return best_id if best_iou >= 0.6 else None

    # ------------------------------------------------------------------
    def _scan_session_state(self) -> Dict[str, Any]:
        return {
            "active": self._auto_scan_active,
            "running": self._auto_scan_inflight,
            "target_id": self._auto_scan_target_id,
        }

    # ------------------------------------------------------------------
    def _broadcast_scan_session(self) -> None:
        self.broadcast.broadcast({"type": "scan_session", **self._scan_session_state()})

    # ------------------------------------------------------------------
    def _pick_auto_scan_candidate(self, now: float) -> Optional[str]:
        with self._frame_lock:
            fh, fw = self._latest_frame.shape[:2] if self._latest_frame is not None else (720, 1280)
        candidates: List[Tuple[float, str]] = []
        for victim in self.scene.all_victims():
            if victim.off_screen:
                continue
            if victim.scans:
                continue
            if now - float(victim.last_seen or 0.0) > 1.0:
                continue
            if now - float(self._auto_scan_last_completed.get(victim.id, 0.0)) < 5.0:
                continue
            x1, y1, x2, y2 = victim.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            center_penalty = abs(cx - (fw * 0.5)) + abs(cy - (fh * 0.5)) * 0.6
            area_bonus = float(w * h) * 0.001
            score = center_penalty - area_bonus
            candidates.append((score, victim.id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    # ------------------------------------------------------------------
    def _maybe_start_auto_scan(self, now: float) -> None:
        if not self._auto_scan_active or self._auto_scan_inflight:
            return
        vid = self._pick_auto_scan_candidate(now)
        if not vid:
            if self._auto_scan_target_id is not None:
                self._auto_scan_target_id = None
                self._broadcast_scan_session()
            return
        self._auto_scan_inflight = True
        self._auto_scan_target_id = vid
        self._broadcast_scan_session()
        threading.Thread(
            target=self._run_scan,
            args=(vid, "auto"),
            daemon=True,
            name=f"auto-scan-{vid}",
        ).start()

    # ------------------------------------------------------------------
    def _update_victim_live(self,
                             victim: Victim,
                             frame: np.ndarray,
                             tracked,
                             now: float,
                             transcript: str,
                             rppg_results: Dict[str, Any]) -> None:
        """Continuous-update path (mode == 'live'). Unchanged from previous
        behaviour aside from passing a precomputed silhouette in if we have one.
        """
        if now - self._last_wound_scan.get(victim.id, 0.0) > self._wound_scan_interval:
            try:
                wounds, blood = self.wounds.process(frame, tracked.bbox)
                for w in wounds:
                    w.body_location = self.body.locate(w.bbox, tracked.bbox, tracked.keypoints)
                victim.wound_regions = wounds
                victim.blood_regions = blood
                if any("tourniquet" in (w.label or "").lower() for w in wounds):
                    self._maybe_auto_start_tq(victim, source="detection",
                                              note="TQ detected live")
            except Exception as exc:
                print(f"[main] live wound scan failed ({exc})")
            self._last_wound_scan[victim.id] = now

        est = rppg_results.get(tracked.track_id)
        if est:
            victim.vitals.hr = est.hr
            victim.vitals.rr = est.rr
            victim.vitals.hr_confidence = est.hr_confidence
            victim.vitals.rr_confidence = est.rr_confidence
            victim.vitals.last_updated = est.timestamp or now

        victim.transcript = transcript
        victim.transcript_updated = now
        codewords = [c["codeword"] for c in scan_transcript(transcript)]
        victim.tccc_codewords = codewords
        if any("TQ applied" == c for c in codewords):
            self._maybe_auto_start_tq(victim, source="codeword",
                                      note="transcript: TQ applied")

        scen = self.scenarios[self.scene.scenario]
        if scen.get("estimate_burn_percent"):
            victim.tbsa_burn_percent = estimate_tbsa_percent(victim.wound_regions)
        else:
            victim.tbsa_burn_percent = None

        march_state = derive_march(victim, scen)
        victim.march = march_state.to_dict()

        if not victim.salt_tag_confirmed:
            suggestion = suggest_salt(victim, march_state)
            if suggestion.tag in {SaltTag.GREY, SaltTag.BLACK}:
                victim.salt_tag = SaltTag.UNTAGGED
            else:
                victim.salt_tag = suggestion.tag
            victim.salt_tag_reason = suggestion.reason

    # ------------------------------------------------------------------
    def _synth_geo(self, bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> Tuple[float, float]:
        h, w = shape
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        anchor_lat = self.cfg["atak"].get("anchor_lat", 38.8895)
        anchor_lon = self.cfg["atak"].get("anchor_lon", -77.0353)
        lat = anchor_lat + (0.5 - cy / max(1, h)) * 0.0004
        lon = anchor_lon + (cx / max(1, w) - 0.5) * 0.0005
        return lat, lon

    # ------------------------------------------------------------------
    def _render_preview(self, frame: np.ndarray) -> None:
        import cv2

        overlay = frame.copy()
        active_victims = [v for v in self.scene.all_victims() if not v.off_screen]
        archived_count = max(0, len(self.scene.victims) - len(active_victims))
        for v in active_victims:
            x1, y1, x2, y2 = v.bbox
            color_hex = {
                SaltTag.RED: (60, 60, 229),
                SaltTag.YELLOW: (45, 192, 251),
                SaltTag.GREEN: (71, 160, 67),
                SaltTag.GREY: (158, 158, 158),
                SaltTag.BLACK: (33, 33, 33),
                SaltTag.UNTAGGED: (100, 100, 100),
            }[v.salt_tag]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_hex, 2)
            label = f"{v.id}  {v.salt_tag.value}"
            if v.priority and v.priority != "P5":
                label += f"  {v.priority}"
            if v.vitals.hr:
                label += f"  HR {v.vitals.hr:.0f}"
            cv2.putText(overlay, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_hex, 2)
            wound_src = v.wound_regions
            blood_src = v.blood_regions
            hud_prev = self._preview_wound_overlay.get(v.id)
            if self.mode == "scan" and (not wound_src) and hud_prev:
                wound_src, blood_src = hud_prev[0], hud_prev[1]
                cv2.putText(overlay, "HUD preview", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 255), 1)
            for w in wound_src:
                wx1, wy1, wx2, wy2 = w.bbox
                cv2.rectangle(overlay, (wx1, wy1), (wx2, wy2), (0, 140, 255), 1)
                cv2.putText(overlay, w.label, (wx1, max(12, wy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
            for b in blood_src:
                bx1, by1, bx2, by2 = b.bbox
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 220), 1)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, overlay)
        hud_hint = ""
        if self.mode == "scan" and self._scan_wound_preview_enabled:
            hud_hint = "  | orange=HUD wound preview (3s/casualty)"
        victim_status = f"victims={len(active_victims)}"
        if archived_count:
            victim_status += f" archived={archived_count}"
        cv2.putText(overlay,
                    f"MASCAL [{self.scene.scenario}]  mode={self.mode}  "
                    f"{victim_status}  frame={self._frame_idx}"
                    f"{hud_hint}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self._preview_disabled:
            return
        try:
            cv2.imshow("MASCAL Triage Relay — edge node", overlay)
        except cv2.error as exc:
            # Typically "The function is not implemented" when
            # opencv-python-headless is installed (e.g. pulled in by
            # insightface). Degrade gracefully; the dashboard remains the
            # primary UI.
            self._preview_disabled = True
            print(f"[main] cv2.imshow unavailable ({exc}); preview window disabled. "
                  f"Dashboard at http://localhost:8080/ still works. "
                  f"Fix: pip install --force-reinstall opencv-python && "
                  f"pip uninstall -y opencv-python-headless")

    # ------------------------------------------------------------------
    async def _on_control(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "confirm_tag":
            vid = msg.get("victim_id")
            tag = msg.get("tag")
            actor = msg.get("actor", "medic")
            victim = self.scene.get(vid) if vid else None
            if victim is None:
                return
            try:
                new_tag = SaltTag(tag)
            except ValueError:
                return
            prev = victim.salt_tag.value
            victim.salt_tag = new_tag
            victim.salt_tag_confirmed = True
            victim.priority = derive_priority(new_tag.value, victim.wound_regions, victim.march)
            self.audit.write("tag_confirmed", actor=actor, victim_id=vid,
                             previous_state=prev, new_state=new_tag.value,
                             payload={"reason": victim.salt_tag_reason})
            self.atak.publish(victim, force=True)
            self.broadcast.broadcast({"type": "audit",
                                      "event": {"kind": "tag_confirmed", "victim_id": vid,
                                                "tag": new_tag.value, "actor": actor}})
        elif mtype == "set_scenario":
            sc = msg.get("scenario")
            if sc in self.scenarios:
                prev = self.scene.scenario
                self.scene.scenario = sc
                with self._pipeline_lock:
                    self.wounds.set_prompt(self.scenarios[sc]["gdino_prompts"])
                self.audit.write("scenario_changed", actor=msg.get("actor", "medic"),
                                 previous_state=prev, new_state=sc)
        elif mtype == "set_mode":
            new_mode = msg.get("mode", "scan")
            if new_mode not in ("scan", "live"):
                return
            prev = self.mode
            self.mode = new_mode
            self.audit.write("mode_changed", actor=msg.get("actor", "medic"),
                             previous_state=prev, new_state=new_mode)
            self.broadcast.broadcast({"type": "mode", "mode": new_mode})
        elif mtype == "set_profile":
            new_profile = msg.get("profile")
            if new_profile in ("fast", "balanced", "max"):
                threading.Thread(
                    target=self._reload_profile,
                    args=(new_profile, msg.get("actor", "medic")),
                    daemon=True,
                ).start()
        elif mtype == "start_scan":
            vid = msg.get("victim_id")
            actor = msg.get("actor", "medic")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_scan, vid, actor)
        elif mtype == "start_auto_scan":
            self.mode = "scan"
            self._auto_scan_active = True
            self.broadcast.broadcast({"type": "mode", "mode": self.mode})
            self._broadcast_scan_session()
            self.audit.write("auto_scan_started", actor=msg.get("actor", "medic"))
        elif mtype == "stop_auto_scan":
            self._auto_scan_active = False
            self._auto_scan_target_id = None
            self._broadcast_scan_session()
            self.audit.write("auto_scan_stopped", actor=msg.get("actor", "medic"))
        elif mtype in ("confirm_wound", "reject_wound"):
            self._apply_wound_confirmation(
                vid=msg.get("victim_id"),
                scan_id=msg.get("scan_id"),
                wound_idx=msg.get("wound_idx"),
                decision="confirmed" if mtype == "confirm_wound" else "rejected",
                actor=msg.get("actor", "medic"),
            )
        elif mtype == "generate_mist":
            vid = msg.get("victim_id")
            victim = self.scene.get(vid) if vid else None
            if victim is None:
                return
            loop = asyncio.get_event_loop()
            card = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(victim, _rehydrate_march(victim.march),
                                           self.scenarios[self.scene.scenario]),
            )
            mist_dict = card.to_dict()
            # DD-1380-shaped handoff bundle attached to the MIST card.
            mist_dict["dd1380"] = self._build_dd1380(victim, mist_dict)
            victim.mist = mist_dict
            self.audit.write("mist_generated", actor=msg.get("actor", "medic"),
                             victim_id=vid, new_state=victim.mist)
            self.broadcast.broadcast({"type": "mist", "victim_id": vid, "mist": victim.mist})
            self.atak.publish(victim, force=True)
        elif mtype == "note":
            vid = msg.get("victim_id")
            text = (msg.get("text") or "").strip()
            if not text:
                return
            if self.audio is not None:
                self.audio.push_text(text)
            if vid:
                victim = self.scene.get(vid)
                if victim:
                    victim.transcript = (victim.transcript + " " + text).strip()[-800:]
                    victim.transcript_updated = time.time()
                    self.audit.write("note_added", actor=msg.get("actor", "medic"),
                                     victim_id=vid, payload={"text": text})
        elif mtype == "start_timer":
            vid = msg.get("victim_id")
            kind = msg.get("kind", "tourniquet")
            duration = float(msg.get("duration_seconds", 7200.0))
            victim = self.scene.get(vid) if vid else None
            if victim is None:
                return
            from state.victim import InterventionTimer
            victim.timers.append(InterventionTimer(kind=kind, started_at=time.time(),
                                                    duration_seconds=duration,
                                                    note=msg.get("note", "")))
            self.audit.write("timer_started", actor=msg.get("actor", "medic"),
                             victim_id=vid, payload={"kind": kind, "duration": duration})

    # ------------------------------------------------------------------
    def _maybe_auto_start_tq(self, victim: Victim, source: str, note: str = "") -> bool:
        """Auto-start a 2-hour tourniquet countdown if not already running.

        TCCC teaches that any tourniquet application should be timestamped
        immediately — the 2-hour ischemia clock is a hard clinical
        threshold. We trigger from two signals:

        * ``source == "detection"`` — wound segmenter detected a
          tourniquet/strap on the victim.
        * ``source == "codeword"``  — TCCC scanner matched "TQ ON" /
          "tourniquet applied" in the transcript.

        We never double-start: if a tourniquet timer is already running
        (even one the medic started manually) we bail out.  Returns True
        if a new timer was started.
        """
        from state.victim import InterventionTimer
        for t in victim.timers or []:
            if (t.kind or "").lower().startswith("tourniquet"):
                return False
        timer = InterventionTimer(
            kind="tourniquet",
            started_at=time.time(),
            duration_seconds=7200.0,
            note=note or f"auto-start ({source})",
            auto=True,
            source=source,
        )
        victim.timers.append(timer)
        self.audit.write(
            "timer_started", actor="ai",
            victim_id=victim.id,
            payload={"kind": "tourniquet", "duration": 7200.0,
                     "auto": True, "source": source},
        )
        print(f"[main] auto-started TQ timer for {victim.id} (source={source})", flush=True)
        try:
            self.broadcast.broadcast({
                "type": "timer_started",
                "victim_id": victim.id,
                "kind": "tourniquet",
                "duration_seconds": 7200.0,
                "auto": True,
                "source": source,
                "ts": time.time(),
            })
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # TCCC timer milestones: 60M reassess, 90M prep convert, 2H breach.
    # Bumping the threshold here also requires updating the matching CSS
    # selectors in dashboard/styles.css (timer-chip states).
    _TQ_MILESTONES = [
        ("60m", 3600.0, "tourniquet 60 min — reassess bleeding / distal pulses"),
        ("90m", 5400.0, "tourniquet 90 min — prepare to convert if safe"),
        ("2h",  7200.0, "tourniquet 2 h — ischemia risk, vascular assessment"),
    ]

    def _check_timer_milestones(self, now: float) -> None:
        """Emit a one-shot alert for each TQ milestone as it's crossed.

        We walk all victims each frame (cheap; o(victims * timers)) and
        compare ``elapsed = now - started_at`` against the milestone
        thresholds.  The per-timer ``alerted`` list stops duplicate
        notifications.
        """
        for victim in list(self.scene.victims.values()):
            for timer in list(victim.timers or []):
                if not (timer.kind or "").lower().startswith("tourniquet"):
                    continue
                elapsed = now - float(timer.started_at or now)
                for tag, threshold, message in self._TQ_MILESTONES:
                    if elapsed >= threshold and tag not in timer.alerted:
                        timer.alerted.append(tag)
                        self.audit.write(
                            "timer_milestone", actor="ai",
                            victim_id=victim.id,
                            payload={"kind": timer.kind, "milestone": tag,
                                     "elapsed_seconds": elapsed,
                                     "message": message},
                        )
                        try:
                            self.broadcast.broadcast({
                                "type": "timer_milestone",
                                "victim_id": victim.id,
                                "kind": timer.kind,
                                "milestone": tag,
                                "elapsed_seconds": elapsed,
                                "message": message,
                                "ts": now,
                            })
                        except Exception:
                            pass

    # ------------------------------------------------------------------
    def _run_scan(self, victim_id: str, actor: str) -> None:
        """Run a ScanEngine.capture for one victim and merge the result in.

        Samples additional frames across a short sweep window so the result
        feels like a head-to-toe body scan: the engine unions wound
        detections across frames and picks the richest frame as the
        canonical crop/face reference. Progress ticks are broadcast as
        ``scan_progress`` WS events.
        """
        if not self._scan_run_lock.acquire(blocking=False):
            self.broadcast.broadcast({"type": "scan_error", "victim_id": victim_id,
                                       "reason": "scan already in progress"})
            return
        self._auto_scan_inflight = True
        self._auto_scan_target_id = victim_id
        self._broadcast_scan_session()
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        victim = self.scene.get(victim_id) if victim_id else None
        if frame is None or victim is None:
            self.broadcast.broadcast({"type": "scan_error", "victim_id": victim_id,
                                       "reason": "no frame or victim"})
            self._auto_scan_inflight = False
            if self._auto_scan_target_id == victim_id:
                self._auto_scan_target_id = None
            self._broadcast_scan_session()
            self._scan_run_lock.release()
            return

        def _frame_provider() -> Optional[np.ndarray]:
            with self._frame_lock:
                if self._latest_frame is None:
                    return None
                return self._latest_frame.copy()

        # Slightly longer sweep: 5 samples over ~1.2s improves the odds of
        # capturing one stable/sharp frame for wound localization, which is
        # especially important for handheld demos and monitor-recorded input.
        sweep_samples = 5
        def _progress(step: int, total: int, phase: str) -> None:
            try:
                self.broadcast.broadcast({
                    "type": "scan_progress",
                    "victim_id": victim_id,
                    "step": int(step),
                    "total": int(total),
                    "phase": phase,
                })
            except Exception:
                pass

        # Tell the UI whether this is a fresh scan or a re-scan of a known
        # victim. The UI uses this to flash "Recognized as Echo-3 · adding
        # scan #2" so the medic knows the face-ID already linked them.
        prior_scans = len(victim.scans)
        if prior_scans > 0:
            self.broadcast.broadcast({
                "type": "scan_recognized",
                "victim_id": victim_id,
                "prior_scan_count": prior_scans,
                "ts": time.time(),
            })

        self.broadcast.broadcast({
            "type": "scan_progress",
            "victim_id": victim_id,
            "step": 0,
            "total": sweep_samples,
            "phase": "starting",
        })

        try:
            with self._pipeline_lock:
                record, extras = self.scan_engine.capture(
                    frame_bgr=frame,
                    victim_bbox=victim.bbox,
                    keypoints=victim.keypoints,
                    victim_id=victim_id,
                    scenario_id=self.scene.scenario,
                    transcript_snippet=victim.transcript or self.scene.global_transcript,
                    frame_provider=_frame_provider,
                    sweep_duration_sec=1.2,
                    sweep_samples=sweep_samples,
                    progress_cb=_progress,
                )
        except Exception as exc:
            print(f"[main] scan failed ({exc})")
            self.broadcast.broadcast({"type": "scan_error", "victim_id": victim_id,
                                       "reason": str(exc)})
            self._auto_scan_inflight = False
            if self._auto_scan_target_id == victim_id:
                self._auto_scan_target_id = None
            self._broadcast_scan_session()
            self._scan_run_lock.release()
            return

        # Merge extras into the live victim.
        victim.wound_regions = extras.get("wound_regions", victim.wound_regions)
        victim.blood_regions = extras.get("blood_regions", victim.blood_regions)
        victim.march = extras.get("march", victim.march)
        vitals = extras.get("vitals") or {}
        if vitals.get("hr") is not None:
            victim.vitals.hr = vitals.get("hr")
        if vitals.get("rr") is not None:
            victim.vitals.rr = vitals.get("rr")
        if not victim.salt_tag_confirmed:
            tag = extras.get("salt_tag")
            if tag is not None:
                victim.salt_tag = tag
            victim.salt_tag_reason = extras.get("salt_reason", victim.salt_tag_reason)
        victim.tbsa_burn_percent = extras.get("tbsa_burn_percent", victim.tbsa_burn_percent)
        # Keywords bundle already includes codewords + injury shorthand;
        # fall back to raw codewords when the scan produced no keywords.
        scan_keywords = list(extras.get("keywords") or [])
        if not scan_keywords:
            scan_keywords = list(extras.get("tccc_codewords") or [])
        if extras.get("face_embedding"):
            victim.face_embedding = extras["face_embedding"]
        if extras.get("face_thumb_url"):
            victim.face_thumb_url = extras["face_thumb_url"]
        victim.priority = extras.get("priority", victim.priority)
        # Merge keywords so the chip list stays sticky across scans (useful
        # when a subsequent sweep misses a finding that was captured before).
        existing = set(k.lower() for k in (victim.tccc_codewords or []))
        for kw in scan_keywords:
            if kw.lower() not in existing:
                victim.tccc_codewords.append(kw)
                existing.add(kw.lower())
        # Number this scan sequentially for the victim (re-scans increment).
        record.scan_index = prior_scans + 1
        victim.scans.append(record)
        victim.last_scan_id = record.scan_id
        victim.total_scan_count = prior_scans + 1

        # Auto-start a TQ countdown if the scan detected a tourniquet.
        # The medic can override from the tile if it was a false positive.
        if any(
            "tourniquet" in (getattr(w, "label", "") or "").lower()
            and (getattr(w, "confirmation", "pending") or "pending") != "rejected"
            for w in (victim.wound_regions or [])
        ):
            self._maybe_auto_start_tq(victim, source="detection",
                                      note="TQ detected by scan")

        self.audit.write("scan_captured", actor=actor, victim_id=victim_id,
                         payload={"scan_id": record.scan_id,
                                   "priority": record.priority,
                                   "scan_index": record.scan_index})
        self.broadcast.broadcast({
            "type": "scan_ready",
            "victim_id": victim_id,
            "scan": asdict(record),
            "scan_index": record.scan_index,
            "total_scans": victim.total_scan_count,
        })
        self._auto_scan_last_completed[victim_id] = time.time()
        self._auto_scan_inflight = False
        if self._auto_scan_target_id == victim_id:
            self._auto_scan_target_id = None
        self._broadcast_scan_session()
        self._scan_run_lock.release()

    # ------------------------------------------------------------------
    def _apply_wound_confirmation(self, vid: Optional[str], scan_id: Optional[str],
                                  wound_idx, decision: str, actor: str) -> None:
        """Update a wound's confirmation, rebuild priority, teach negatives.

        ``decision`` is ``"confirmed"`` or ``"rejected"``. For rejections we
        also record a session-level negative on the ScanEngine so future
        scans skip the same (label, body_region) detection at the
        segmentation layer — a crucial "medic teaches the model" loop.
        """
        if vid is None or scan_id is None or wound_idx is None:
            return
        try:
            wound_idx = int(wound_idx)
        except Exception:
            return
        victim = self.scene.get(vid)
        if victim is None:
            return
        scan = next((s for s in victim.scans if s.scan_id == scan_id), None)
        if scan is None or wound_idx < 0 or wound_idx >= len(scan.wounds):
            return

        wound_dict = scan.wounds[wound_idx]
        wound_dict["confirmation"] = decision
        label = wound_dict.get("label", "")
        region = wound_dict.get("body_region", "")

        # Mirror on the live WoundRegion list if this scan is the latest.
        if scan_id == victim.last_scan_id and wound_idx < len(victim.wound_regions):
            victim.wound_regions[wound_idx].confirmation = decision

        if decision == "rejected":
            # 1) Pull the rejected wound out of the active priority calculation.
            #    We keep the row in scan.wounds (for audit history) but build
            #    a filtered list for priority.
            active_wounds = []
            for wd in victim.wound_regions:
                if (getattr(wd, "confirmation", "pending") or "pending") == "rejected":
                    continue
                active_wounds.append(wd)
            victim.priority = derive_priority(
                victim.salt_tag.value, active_wounds, victim.march,
            )
            # 2) Teach the segmenter: don't re-surface this label for this
            #    victim next scan.
            self.scan_engine.add_session_negative(vid, label, region)
            pair = (label.lower(), (region or "").lower())
            if pair not in [(l.lower(), r.lower()) for l, r in victim.rejected_findings]:
                victim.rejected_findings.append(pair)
            scan.priority = victim.priority
            # If the victim now has no active wounds, set keywords to reflect.
            victim.tccc_codewords = [
                kw for kw in (victim.tccc_codewords or [])
                if label.lower() not in kw.lower()
            ]

        self.audit.write(
            "scan_confirmed" if decision == "confirmed" else "scan_rejected",
            actor=actor,
            victim_id=vid,
            payload={"scan_id": scan_id, "wound_idx": wound_idx,
                      "label": label, "body_region": region,
                      "decision": decision},
        )
        self.broadcast.broadcast({
            "type": "scan_confirmed" if decision == "confirmed" else "scan_rejected",
            "victim_id": vid,
            "scan_id": scan_id,
            "wound_idx": wound_idx,
            "decision": decision,
            "priority": victim.priority,
        })

    # ------------------------------------------------------------------
    def _build_dd1380(self, victim: Victim, mist: Dict[str, Any]) -> Dict[str, Any]:
        """DD Form 1380 (Tactical Combat Casualty Care Card) JSON.

        This is a condensed, JSON-first representation suitable for EMR /
        ATAK handoff. Fields correspond to the paper card sections.
        """
        scan_id = victim.last_scan_id
        wounds = [
            {
                "label": w.label,
                "body_region": w.body_location,
                "severity": w.severity,
            }
            for w in victim.wound_regions
        ]
        return {
            "casualty_id": victim.id,
            "priority": victim.priority,
            "mechanism": mist.get("mechanism"),
            "injuries_sustained": wounds,
            "vitals": {
                "hr": victim.vitals.hr, "rr": victim.vitals.rr, "spo2": victim.vitals.spo2,
            },
            "treatments": mist.get("treatment", []),
            "tccc_codewords": victim.tccc_codewords,
            "scan_id": scan_id,
            "scan_frame_url": f"/api/scans/{scan_id}/frame.jpg" if scan_id else None,
            "scan_crop_url": f"/api/scans/{scan_id}/crop.jpg" if scan_id else None,
            "timestamp": time.time(),
        }


def _rehydrate_march(march_dict: Dict[str, Any]) -> MarchState:
    """Rebuild a MarchState from its to_dict() form (for LLM prompting)."""
    from state.march import MarchField, Status

    def _mf(d: Dict[str, Any]) -> MarchField:
        return MarchField(
            status=Status(d.get("status", "unknown")),
            confidence=float(d.get("confidence", 0.0)),
            reason=str(d.get("reason", "")),
        )

    if not march_dict:
        return MarchState()
    return MarchState(
        massive_hemorrhage=_mf(march_dict.get("M", {})),
        airway=_mf(march_dict.get("A", {})),
        respiration=_mf(march_dict.get("R", {})),
        circulation=_mf(march_dict.get("C", {})),
        head_hypothermia=_mf(march_dict.get("H", {})),
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    args = parse_args()
    print("[main] MASCAL edge node starting…", flush=True)
    node = EdgeNode(args)
    print("[main] pipeline ready; entering capture loop.", flush=True)
    node.run()


if __name__ == "__main__":
    main()
