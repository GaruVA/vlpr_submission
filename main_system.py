"""
main_system.py
Production Pipeline — Full Dockyard RTSP Deployment

Sprint 3 Additions:
  - CLAHE preprocessing wired into the character recognition pipeline.
    enhance_plate_contrast() from src/utils.py is now called on every plate
    crop immediately before char_model.predict(). Fulfils Interim Report
    Chapter 07 Snippet 2 and §7.4 Challenge 1 resolution.

  - SEC-002: Hardcoded RTSP credentials removed from source code.
    CONFIG['VIDEO_SOURCE'] now reads from the RTSP_URL environment variable.
    Falls back to 'rtsp://localhost' if the variable is not set.

  - SEC-001: Pre-flight plate format validation added before any call to
    insert_plate_detection(). Only canonical SL-format plates ('XX-NNNN'
    or 'XXX-NNNN') may progress to the persistence layer.

  - Sprint 1 cleanup: the tempfile module dependency has been fully removed
    (superseded by the BUG-002 fix that passes numpy arrays directly to YOLO).

Sprint 2 Additions (retained):
  - CameraStream: Background RTSP capture thread.
    Fulfils the CameraStream class from Interim Report §5.2 class diagram:
    "uses a background thread to continuously pull frames from the camera's
    RTSP feed into a shared buffer. The main processing thread always reads
    the most recent frame from this buffer rather than waiting for the
    capture operation, which prevents processing lag."

    Without this class:
      cap.read() is blocking. The main thread stalls waiting for a new
      frame while the GPU sits idle. The next frame read is then stale
      by (inference_time + OCR_time), typically 12–50ms. Over a 1.5s
      budget this compounds visibly.
    With CameraStream:
      A daemon thread reads continuously into a 1-slot buffer. The main
      thread always gets the latest available frame in O(1) time.

  - LPM-MLED wired into the save decision:
    Before committing a plate to the database, find_best_match() is called
    to confirm the OCR string matches a registered vehicle (or is a
    plausible OCR variant of one) via the Weighted Homoglyph algorithm.
    This fulfils FR-03 from Interim Report §3.1.

  - log_fraud_event() wired into the spatial verification block:
    When the STC Engine raises a fraud alert, the FraudAlert dataclass
    is retrieved and persisted to SQLite via DatabaseManager.log_fraud_event().
    This fulfils FR-05 and the FraudEvent entity from Interim Report §5.3.

  - gate_id passed to insert_plate_detection():
    GateID is now recorded in the access_log table as required by the ER
    Diagram (Interim Report §5.3).

Sprint 1 Retained:
  - BUG-001/002: tempfile dependency eliminated; numpy array passed directly
    to char_model.predict() (no temp file I/O).
"""

from __future__ import annotations

import os
import re
import sys
import time
import threading
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO

from src.tracker   import PlateTracker
from src.validator import SriLankanPlateValidator, is_reasonable_plate_text
from src.database  import DatabaseManager
from src.spatial   import SpatialVerifier
from src.utils     import smart_character_ordering, enhance_plate_contrast

# ── SEC-001: Canonical Sri Lankan plate format regex ─────────────────────────
# Used as a pre-flight validation gate before any data is submitted to the
# persistence layer. Rejects garbage OCR output at the earliest possible point.
# Accepts: 'WP-1234' (2-letter) and 'CAB-1234' (3-letter) formats.
_SL_PLATE_RE = re.compile(r'^[A-Z]{2,3}-\d{4}$')

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

CONFIG: dict = {
    'PLATE_MODEL_PATH':            os.path.join(MODELS_DIR, 'plate_detection.pt'),
    'CHAR_MODEL_PATH':             os.path.join(MODELS_DIR, 'character_recognition.pt'),
    # SEC-002 FIX: RTSP credentials removed from source code.
    # Set the RTSP_URL environment variable before running:
    #   export RTSP_URL="rtsp://user:pass@192.168.100.132:554/Streaming/Channels/101"
    # Falls back to a safe localhost placeholder if the variable is not set.
    # This prevents credentials appearing in version control, logs, or
    # stack traces.
    'VIDEO_SOURCE': os.environ.get(
        'RTSP_URL',
        'rtsp://localhost:554/stream'
    ),
    'CONFIDENCE_THRESHOLD':        0.5,
    'TRACKER_MAX_AGE':             30,
    'TRACKER_MIN_HITS':            3,
    'TRACKER_IOU_THRESHOLD':       0.3,
    'FRAME_SKIP':                  3,
    'SAVE_DETECTIONS':             True,
    'SHOW_INDIVIDUAL_PLATES':      True,
    'DATABASE_ENABLED':            True,
    'API_BASE_URL':                "https://esystems.cdl.lk/backend-Test/NPRCamera/RFID",
    'DEVICE':                      '01',
    'IN_OUT':                      'I',
    'DB_ASYNC_MODE':               True,
    'MIN_CONFIDENCE_FOR_DB':       0.75,
    'MIN_HITS_FOR_DB':             40,
    'MODE':                        'PRODUCTION',
    'GATE_ID':                     'GATE1',
    'ENABLE_SPATIAL_VERIFICATION': True,
    'SPATIAL_TRAVEL_TIMES': {
        ('GATE1', 'GATE2'): 120,
        ('GATE1', 'GATE3'): 300,
        ('GATE2', 'GATE3'): 180,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CameraStream — Background RTSP Capture Thread
#
# Fulfils the CameraStream class from Interim Report §5.2 class diagram:
#   -rtsp_url: String
#   -frame_buffer: Buffer
#   +start_stream()
#   +get_latest_frame(): Image
# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    Decoupled RTSP frame capture using a background daemon thread.

    Problem solved (ARCH-001):
      Without this class, cap.read() is a blocking call. Every frame the
      main thread calls cap.read(), the OS network stack fills a buffer,
      decodes a JPEG/H264 frame, and returns it. During this wait (~5–30ms
      for RTSP), the GPU is idle. The returned frame is already stale by
      the time inference begins.

    Solution:
      A background thread calls cap.read() in a tight loop and always
      overwrites _frame with the latest result. The main thread calls
      get_latest_frame() which is a O(1) memory copy under a lock — it
      never blocks on network I/O again.

    Thread safety:
      _frame is protected by _lock. The background thread holds the lock
      only during the np.ndarray assignment (microseconds). The main thread
      holds it only during frame.copy() (microseconds). They never contend
      for meaningful duration.

    Interim Report §5.2 CameraStream class diagram fields:
      -rtsp_url   → self.source
      -frame_buffer → self._frame (single-frame buffer)
      +start_stream() → __init__ (thread starts at construction)
      +get_latest_frame() → get_latest_frame()
    """

    def __init__(self, source: str) -> None:
        self.source  = source
        # Integer index required for local webcams; keep string for URLs/paths.
        _cap_source  = int(source) if source.lstrip('-').isdigit() else source
        self.cap     = cv2.VideoCapture(_cap_source, cv2.CAP_MSMF)

        if source.startswith(('rtsp://', 'http://')):
            # Minimise OS-level buffer depth — we only ever want the latest frame.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame:   np.ndarray | None = None
        self._ret:     bool              = False
        self._lock:    threading.Lock    = threading.Lock()
        self._running: bool              = True

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture",
        )
        self._thread.start()

        # Block the constructor until the first frame is available,
        # so the caller can immediately call get_latest_frame() safely.
        deadline = time.time() + 10.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)

        if self._frame is None:
            print(f"⚠️  CameraStream: no frame received within 10s for {source}")

    def _capture_loop(self) -> None:
        """
        Tight read loop — always keeps _frame current.
        This is the 'continuously pull frames into a shared buffer' behaviour
        described in Interim Report §5.2.
        """
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._ret   = ret
            else:
                # Brief sleep on read failure avoids 100% CPU spin
                # on a dead RTSP connection.
                time.sleep(0.01)

    def get_latest_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Return (ret, frame) — the most recently captured frame.

        The returned frame is a copy, preventing the background thread from
        mutating the array while the main thread is processing it.

        Returns (False, None) if no frame has been received yet.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def stop(self) -> None:
        """Signal the background thread to exit and release the capture device."""
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main Detection Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_enhanced_plate_detection() -> None:
    """
    Full production detection pipeline with all three layers active.

    Layer 1: YOLOv8 plate detection + YOLO character recognition.
    Layer 2: Positional correction + LPM-MLED registered vehicle lookup.
    Layer 3: Spatial-Temporal Correlation Engine (fraud detection).

    Persistence: SQLite audit log (local) + CDL REST API (async).
    """
    if not os.path.exists(CONFIG['PLATE_MODEL_PATH']) or \
       not os.path.exists(CONFIG['CHAR_MODEL_PATH']):
        print("❌ Model files not found.")
        print(f"   Plate model: {CONFIG['PLATE_MODEL_PATH']}")
        print(f"   Char model:  {CONFIG['CHAR_MODEL_PATH']}")
        return

    source = CONFIG['VIDEO_SOURCE']
    print("🚗  Enhanced Vehicle License Plate Detection System")
    print("=" * 65)
    print(f"  📹 Video Source:  {source}")
    print(f"  🎯 Plate Model:   {os.path.basename(CONFIG['PLATE_MODEL_PATH'])}")
    print(f"  🔤 Char Model:    {os.path.basename(CONFIG['CHAR_MODEL_PATH'])}")
    if CONFIG.get('DATABASE_ENABLED'):
        print(
            f"  🗄️  Database:     ENABLED | "
            f"Camera: {CONFIG['DEVICE']} | "
            f"Direction: {CONFIG['IN_OUT']}"
        )
    else:
        print("  🗄️  Database:     DISABLED")
    print("=" * 65)

    # ── Model loading ──────────────────────────────────────────────────────
    try:
        plate_model = YOLO(CONFIG['PLATE_MODEL_PATH'])
        char_model  = YOLO(CONFIG['CHAR_MODEL_PATH'])
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # ── Output directories ─────────────────────────────────────────────────
    for subdir in ("detects", "detects/crops", "detects/raw", "detects/annotated"):
        path = os.path.join(SCRIPT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
    detects_dir = os.path.join(SCRIPT_DIR, "detects")

    # ── Component initialisation ───────────────────────────────────────────
    db_manager: DatabaseManager | None = None
    if CONFIG.get('DATABASE_ENABLED', False):
        try:
            db_manager = DatabaseManager(
                base_url       = CONFIG['API_BASE_URL'],
                cam_code       = CONFIG['DEVICE'],
                device         = CONFIG['DEVICE'],
                in_out         = CONFIG['IN_OUT'],
            )
        except Exception as e:
            print(f"⚠️  Database init failed: {e}")

    tracker   = PlateTracker(
        max_age       = CONFIG['TRACKER_MAX_AGE'],
        min_hits      = CONFIG['TRACKER_MIN_HITS'],
        iou_threshold = CONFIG['TRACKER_IOU_THRESHOLD'],
    )
    validator = SriLankanPlateValidator()

    spatial_verifier: SpatialVerifier | None = None
    if CONFIG.get('ENABLE_SPATIAL_VERIFICATION'):
        spatial_verifier = SpatialVerifier(
            travel_times = CONFIG.get('SPATIAL_TRAVEL_TIMES', {})
        )

    saved_plates: set = set()

    # ── CameraStream — Sprint 2 ARCH-001 fix ──────────────────────────────
    # Replace blocking cap.read() with the background-thread CameraStream.
    # Fulfils CameraStream class from Interim Report §5.2 class diagram.
    print(f"📡 Starting CameraStream for {source}...")
    stream = CameraStream(source)

    frame_count = 0
    plate_windows: dict = {}

    cv2.namedWindow("Enhanced License Plate Detection", cv2.WINDOW_NORMAL)
    print("🎮 Controls: 'q' quit  |  's' save frame")
    print("-" * 50)

    try:
        while stream.is_opened():
            # Get the latest frame from the background buffer — non-blocking.
            ret, frame = stream.get_latest_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Frame skip for performance on RTSP (applies to HTTP too).
            if source.startswith(('rtsp://', 'http://')) and \
               frame_count % CONFIG['FRAME_SKIP'] != 0:
                continue

            # ── Layer 1: Plate Detection ───────────────────────────────────
            try:
                plate_results = plate_model.predict(
                    source=frame, imgsz=640,
                    conf=CONFIG['CONFIDENCE_THRESHOLD'], verbose=False
                )
                plate_res = plate_results[0]
            except Exception as e:
                print(f"⚠️  Plate detection error: {e}")
                continue

            detections: list = []

            if len(plate_res.boxes) > 0:
                for i, box in enumerate(plate_res.boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    confidence = float(plate_res.boxes.conf[i])

                    if confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                        continue

                    # Expand crop slightly for better OCR context.
                    ph, pw = y2 - y1, x2 - x1
                    pad_x = int(pw * 0.1)
                    pad_y = int(ph * 0.1)
                    x1c = max(0, x1 - pad_x);  y1c = max(0, y1 - pad_y)
                    x2c = min(frame.shape[1], x2 + pad_x)
                    y2c = min(frame.shape[0], y2 + pad_y)

                    plate_crop = frame[y1c:y2c, x1c:x2c]
                    if plate_crop.size == 0 or \
                       plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                        continue

                    # Sprint 1 BUG-002 fix: pass numpy array, no temp file.
                    try:
                        # ── Sprint 3 CLAHE Integration ──────────────────────────
                        # Interim Report Chapter 07, Snippet 2 and §7.4 Challenge 1:
                        # enhance_plate_contrast() applies CLAHE in LAB colour space
                        # to restore legible contrast on salt-spray-degraded plates
                        # BEFORE the character recognition model runs.
                        #
                        # This is the preprocessing step described in the report but
                        # previously unwired from the pipeline. Applied here — between
                        # the plate crop and char_model.predict() — so it benefits
                        # OCR accuracy without affecting YOLOv8 plate detection.
                        plate_crop_enhanced = enhance_plate_contrast(plate_crop)
                        # ── End CLAHE Integration ───────────────────────────────

                        char_results = char_model.predict(
                            source=plate_crop_enhanced, imgsz=640, conf=0.3, verbose=False
                        )
                        char_res = char_results[0]

                        plate_text      = "No text detected"
                        char_confidence = 0.0

                        if len(char_res.boxes) > 0:
                            char_boxes: list = []
                            for j, cbox in enumerate(char_res.boxes.xyxy):
                                x1c2, y1c2, x2c2, y2c2 = cbox.cpu().numpy().astype(int)
                                char_conf = float(char_res.boxes.conf[j])
                                cls_id    = int(char_res.boxes.cls[j])
                                if char_conf > 0.3:
                                    char = validator.class_to_char.get(
                                        cls_id, str(cls_id)
                                    )
                                    char_boxes.append({
                                        'x': x1c2, 'y': y1c2,
                                        'w': x2c2 - x1c2, 'h': y2c2 - y1c2,
                                        'char': char, 'conf': char_conf,
                                        'is_letter': char.isalpha(),
                                    })

                            chars, confidences = smart_character_ordering(
                                char_boxes, plate_crop.shape  # original shape (unchanged by CLAHE)
                            )
                            if chars and len(chars) >= 4:
                                plate_text      = ''.join(chars)
                                char_confidence = float(np.mean(confidences)) \
                                    if confidences else 0.0

                        validated_text, validated_conf, is_valid = \
                            validator.validate_and_correct(
                                plate_text, char_confidence
                            )

                        detections.append({
                            'bbox':       (x1, y1, x2, y2),
                            'text':       validated_text,
                            'confidence': validated_conf,
                            'is_valid':   is_valid,
                            'crop':       plate_crop,
                        })

                    except Exception as e:
                        print(f"⚠️  OCR error: {e}")

            # ── Layer 2: Tracking ──────────────────────────────────────────
            tracks = tracker.update(detections)

            # Clean up closed plate sub-windows.
            current_ids = set(tracks.keys())
            for wname in list(plate_windows.keys()):
                if wname.startswith("Plate "):
                    try:
                        if int(wname.split(" ")[1]) not in current_ids:
                            cv2.destroyWindow(wname)
                            del plate_windows[wname]
                    except (ValueError, IndexError):
                        pass

            # ── Layer 2/3: Save decision + Spatial Verification ────────────
            gate_id = CONFIG.get('GATE_ID', 'GATE1')

            for track_id, track in tracks.items():
                if track['hits'] < tracker.min_hits:
                    continue

                _, _, track['is_valid'] = validator.validate_and_correct(
                    track['consensus_text'], track['avg_confidence']
                )

                if not (track.get('is_valid', False)
                        and not track.get('saved_to_db', False)
                        and track['avg_confidence'] >= CONFIG['MIN_CONFIDENCE_FOR_DB']
                        and track['hits'] >= CONFIG['MIN_HITS_FOR_DB']):
                    continue

                plate_text = track['consensus_text']

                # ── Similarity de-duplication ──────────────────────────────
                should_save = True
                for saved_plate in saved_plates:
                    if tracker.calculate_plate_similarity(
                        plate_text, saved_plate
                    ) > 0.85:
                        should_save = False
                        print(
                            f"🔄 Skipping similar plate: {plate_text} "
                            f"(similar to saved: {saved_plate})"
                        )
                        break

                if not should_save:
                    track['saved_to_db'] = True
                    continue

                # ── Layer 2: LPM-MLED Registered Vehicle Lookup ─────────────
                # Fulfils FR-03 (Interim Report §3.1): before committing a
                # detection to the database, confirm it matches a registered
                # vehicle using the Weighted Homoglyph algorithm.
                #
                # If the match fails (no registered plate within distance 0.5),
                # the detection is still logged as DENIED — unregistered vehicle.
                matched_plate, lpm_distance = validator.find_best_match(plate_text)
                is_registered = matched_plate is not None

                if is_registered:
                    print(
                        f"✅ LPM-MLED match: OCR '{plate_text}' → "
                        f"Registered '{matched_plate}' "
                        f"(distance={lpm_distance:.2f})"
                    )
                else:
                    print(
                        f"🔍 LPM-MLED: '{plate_text}' not in registered vehicles "
                        f"(closest distance={lpm_distance:.2f})"
                    )

                # ── SEC-001: Pre-flight format validation ─────────────────────
                # Validate the plate string against the canonical SL format
                # regex before it can reach the persistence layer.
                # This is the pipeline-level counterpart to the same guard
                # in database.py._perform_api_insert() — belt-and-suspenders.
                # Rejects plates that passed OCR but failed normalisation
                # (e.g. an unusually long OCR string, or an all-digit result).
                if not _SL_PLATE_RE.match(plate_text):
                    print(
                        f"🔒 SEC-001 pipeline guard: '{plate_text}' failed "
                        f"format check — skipping persistence."
                    )
                    track['saved_to_db'] = True   # suppress retry
                    continue
                # ── End SEC-001 pipeline guard ────────────────────────────────

                # ── Layer 3: Spatial-Temporal Verification ─────────────────
                # Fulfils FR-04 (Interim Report §3.1) and the STC Engine
                # algorithm described in §6.4.
                is_allowed = True
                decision   = 'GRANTED' if is_registered else 'DENIED'

                if spatial_verifier:
                    is_allowed, stc_reason = spatial_verifier.check_entry(
                        plate_text, gate_id
                    )
                    if not is_allowed:
                        decision = 'DENIED'
                        print(f"🚨 FRAUD ALERT for {plate_text}: {stc_reason}")

                        # Persist the FraudAlert to SQLite — fulfils FR-05
                        # and the FraudEvent entity in Interim Report §5.3.
                        if db_manager:
                            fraud_alert = spatial_verifier.get_latest_fraud_alert()
                            if fraud_alert:
                                db_manager.log_fraud_event(fraud_alert)

                # ── Save crop image ────────────────────────────────────────
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = (
                    f"{track_id}_{plate_text}_"
                    f"{int(track['avg_confidence'] * 100)}_{timestamp}"
                )
                crop_path = os.path.join(
                    detects_dir, "crops", f"crop_{base_name}.jpg"
                )
                cv2.imwrite(crop_path, track['crop'])
                saved_plates.add(plate_text)

                # ── Persist to both channels ───────────────────────────────
                # Channel 1 (SQLite) + Channel 2 (REST API)
                # gate_id now passed through to access_log — fulfils ER Diagram.
                if db_manager and CONFIG.get('DATABASE_ENABLED', False):
                    if CONFIG.get('DB_ASYNC_MODE', True):
                        queued = db_manager.insert_plate_detection(
                            plate_text, track['avg_confidence'],
                            track_id,
                            gate_id=gate_id,
                            decision=decision,
                            sync=False,
                        )
                        track['saved_to_db'] = bool(queued)
                    else:
                        success = db_manager.insert_plate_detection(
                            plate_text, track['avg_confidence'],
                            track_id,
                            gate_id=gate_id,
                            decision=decision,
                            sync=True,
                        )
                        track['saved_to_db'] = bool(success)
                else:
                    track['saved_to_db'] = True

            # ── Draw UI ────────────────────────────────────────────────────
            annotated_frame = frame.copy()
            for track_id, track in tracks.items():
                x1, y1, x2, y2 = track['bbox']
                if track['hits'] >= tracker.min_hits:
                    color = (0, 255, 0) if track.get('is_valid', False) \
                            else (0, 165, 255)
                else:
                    color = (128, 128, 128)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                info_text = (
                    f"ID:{track_id} {track['consensus_text']} "
                    f"({track['avg_confidence']:.2f})"
                )
                cv2.putText(
                    annotated_frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                if (track['hits'] >= tracker.min_hits
                        and CONFIG['SHOW_INDIVIDUAL_PLATES']
                        and 'crop' in track):
                    wname = f"Plate {track_id}"
                    if wname not in plate_windows:
                        cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)
                        plate_windows[wname] = True
                    cv2.imshow(wname, cv2.resize(track['crop'], (300, 100)))

            cv2.imshow("Enhanced License Plate Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(
                    detects_dir, "raw",
                    f"frame_{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
                )
                cv2.imwrite(save_path, annotated_frame)
                print(f"💾 Saved: {save_path}")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        if db_manager:
            db_manager.shutdown()
        print("🛑 Production pipeline stopped.")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="VLPR Production Pipeline")
    _parser.add_argument("--source",    default=None,  help="Camera index (1/2) or RTSP URL")
    _parser.add_argument("--device",    default=None,  help="Camera ID logged to DB (e.g. 01, 02)")
    _parser.add_argument("--direction", default=None,  choices=["I", "O"], help="I=entry O=exit")
    _args = _parser.parse_args()

    if _args.source    is not None: CONFIG['VIDEO_SOURCE'] = _args.source
    if _args.device    is not None: CONFIG['DEVICE']       = _args.device
    if _args.direction is not None: CONFIG['IN_OUT']       = _args.direction

    run_enhanced_plate_detection()
