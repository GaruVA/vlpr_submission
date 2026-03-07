"""
research_demo.py
Dual-Camera Tabletop Research Demo — Viva Defense Script

Layer 3 STC Engine validation using two webcams as Gate A / Gate B,
and 1:18 scale toy vehicles fitted with printed Sri Lankan plates.

Sprint 3 Addition:
  - CLAHE preprocessing wired into the character recognition stage.
    enhance_plate_contrast() is now called on every plate crop immediately
    before char_model.predict(). This is the Interim Report Chapter 07
    Snippet 2 function, previously defined but never called. Under tabletop
    lighting conditions — which often produce flat, low-contrast images on
    printed plate replicas — CLAHE measurably sharpens character boundaries
    and improves OCR stability during the viva demonstration.

Sprint 1 Fix Retained:
  - BUG-003: The guard condition before calling `spatial.check_entry()`
    was `len(p_text) > 4`, which allows the sentinel string "Reading..."
    (9 characters, always > 4) to enter the STC Engine. This caused
    phantom CLONE ATTACK alerts because "Reading..." was stored as a
    vehicle identity in `vehicle_state` and could trigger inter-gate
    violations for noise frames.

    Fix: The guard now uses `is_reasonable_plate_text(p_text)` from
    validator.py, which enforces: minimum length, must contain both
    letters and numbers, no single-character-dominated strings, and
    explicitly rejects known sentinel values.
"""

import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Import refactored modules
from src.tracker import PlateTracker
from src.validator import SriLankanPlateValidator, is_reasonable_plate_text
from src.utils     import enhance_plate_contrast
from src.spatial import SpatialVerifier

# Configuration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')


def run_toy_car_demo():
    """
    Simplified demo for 1:18 scale toy cars with DUAL CAMERAS.

    Camera 1 (Index 1) = GATE A
    Camera 2 (Index 2) = GATE B

    Physics constraint: minimum travel time between any two gates = 5 seconds.
    Any detection transition faster than this threshold triggers a fraud alert,
    demonstrating the Spatial-Temporal Correlation (STC) Engine.
    """
    cam_indices = [1, 2]
    caps = []

    print("🎥 Initializing Dual Camera System...")
    for idx in cam_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"⚠️  Camera {idx} failed to open — falling back to index 0.")
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps.append(cap)

    gate_names = ['GATE A', 'GATE B']

    # ── Model Loading ────────────────────────────────────────────────────────
    plate_model_path = os.path.join(MODELS_DIR, 'plate_detection.pt')
    char_model_path  = os.path.join(MODELS_DIR, 'character_recognition.pt')

    if not os.path.exists(plate_model_path) or not os.path.exists(char_model_path):
        print(f"❌ Models not found in {MODELS_DIR}")
        print(f"   Expected: plate_detection.pt, character_recognition.pt")
        return

    print("🧠 Loading YOLOv8 Models...")
    plate_model = YOLO(plate_model_path)
    char_model  = YOLO(char_model_path)
    print("✅ Models loaded successfully.")

    # ── Component Initialisation ─────────────────────────────────────────────
    # Separate tracker instance per gate — detections are spatially isolated
    # and should not influence each other's IoU matching.
    trackers = [
        PlateTracker(min_hits=1, iou_threshold=0.3),   # Gate A
        PlateTracker(min_hits=1, iou_threshold=0.3),   # Gate B
    ]

    validator = SriLankanPlateValidator()

    # ── Spatial-Temporal Physics Constraints ─────────────────────────────────
    # Tabletop demo rule: any transition between gates A ↔ B must take at
    # least 5 seconds. A detection completing the transition in under 5s
    # is a physics violation (Clone Attack or Speeding).
    spatial = SpatialVerifier({
        ('GATE A', 'GATE B'): 5,
        ('GATE B', 'GATE A'): 5,
    })

    print("\n" + "=" * 65)
    print("🚗  DUAL-CAMERA SPATIAL-TEMPORAL VERIFICATION FRAMEWORK")
    print(f"    Gate A  →  Camera {cam_indices[0]}")
    print(f"    Gate B  →  Camera {cam_indices[1]}")
    print("    Physics Constraint: Min Travel Time = 5.0 seconds")
    print("=" * 65)
    print("    🟩 Green box  = Valid / permitted transition")
    print("    🟥 Red box    = FRAUD detected (Clone / Speeding)")
    print("    🟧 Orange box = Scanning / building consensus")
    print("=" * 65)
    print("Press 'q' to quit\n")

    try:
        while True:
            processed_frames = []

            # ── Per-Camera Processing Loop ───────────────────────────────────
            for i, cap in enumerate(caps):
                ret, frame = cap.read()

                if not ret:
                    # Graceful dropout — show a black frame with a notice.
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "NO SIGNAL", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
                    )
                else:
                    frame = cv2.resize(frame, (640, 480))

                gate_id          = gate_names[i]
                current_tracker  = trackers[i]

                # ── Layer 1: Plate Detection ─────────────────────────────────
                results   = plate_model.predict(frame, conf=0.4, verbose=False)
                plate_res = results[0]
                detections: list = []

                if len(plate_res.boxes) > 0:
                    for j, box in enumerate(plate_res.boxes.xyxy):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)

                        # Clamp to frame boundaries.
                        h, w = frame.shape[:2]
                        x1 = max(0, x1);  y1 = max(0, y1)
                        x2 = min(w, x2);  y2 = min(h, y2)

                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size == 0:
                            continue

                        # ── Layer 1 / Layer 2: Character Recognition ─────────
                        plate_text = ""
                        try:
                            # ── Sprint 3 CLAHE Integration ────────────────────
                            # Interim Report Chapter 07, Snippet 2:
                            # Enhance plate contrast before OCR to handle the
                            # flat, low-contrast images produced by tabletop
                            # lighting on printed plate replicas. CLAHE applies
                            # local histogram equalisation in the LAB colour
                            # space — restoring character edge contrast without
                            # amplifying background noise.
                            plate_crop_enhanced = enhance_plate_contrast(plate_crop)
                            # ── End CLAHE Integration ─────────────────────────

                            # Pass enhanced crop directly — no temp file I/O.
                            char_results = char_model.predict(
                                plate_crop_enhanced, conf=0.4, verbose=False
                            )[0]

                            found_chars: list = []
                            if char_results.boxes:
                                for cbox in char_results.boxes:
                                    cls_id   = int(cbox.cls[0])
                                    x_center = float(cbox.xywh[0][0])
                                    char     = validator.class_to_char.get(cls_id, '?')
                                    found_chars.append((x_center, char))

                                # Sort characters left-to-right by x-centre.
                                found_chars.sort(key=lambda x: x[0])
                                plate_text = ''.join(c[1] for c in found_chars)

                        except Exception:
                            # OCR failures are expected under tabletop lighting —
                            # silently continue rather than crashing the loop.
                            pass

                        # Do NOT assign the "Reading..." sentinel here.
                        # If OCR produced nothing useful, leave plate_text as "".
                        # The is_reasonable_plate_text() guard below will reject it.

                        detections.append({
                            'bbox':       (x1, y1, x2, y2),
                            'text':       plate_text,
                            'confidence': float(plate_res.boxes.conf[j]),
                            'crop':       plate_crop,
                            'is_valid':   True,
                        })

                # ── Layer 2: Tracking & Consensus ────────────────────────────
                tracks = current_tracker.update(detections)

                # Gate label overlay.
                cv2.putText(
                    frame, gate_id, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2
                )

                # ── Layer 3: Spatial-Temporal Verification ───────────────────
                for track_id, track in tracks.items():
                    if track['hits'] < 1:
                        continue

                    x1, y1, x2, y2 = track['bbox']
                    p_text          = track['consensus_text']

                    box_color  = (255, 165, 0)   # Orange — scanning
                    status_msg = ""

                    # ── BUG-003 FIX ──────────────────────────────────────────
                    # BEFORE (WRONG):
                    #   if len(p_text) > 4:
                    #       is_valid, reason = spatial.check_entry(p_text, gate_id)
                    #
                    # The bare length check allowed "Reading..." (9 chars) and
                    # any garbage OCR string of length > 4 to enter the STC
                    # engine, polluting `vehicle_state` with phantom identities
                    # and generating spurious CLONE ATTACK alerts.
                    #
                    # AFTER (CORRECT):
                    #   Use is_reasonable_plate_text() which enforces:
                    #   - 5 ≤ len ≤ 10
                    #   - Must contain both letters and digits
                    #   - No single-character-dominated strings
                    #   - Explicit rejection of known sentinel values
                    # ─────────────────────────────────────────────────────────
                    if is_reasonable_plate_text(p_text):
                        is_valid, reason = spatial.check_entry(p_text, gate_id)

                        if is_valid:
                            box_color  = (0, 255, 0)   # Green — valid
                            status_msg = "OK"
                        else:
                            box_color  = (0, 0, 255)   # Red — fraud
                            status_msg = "FRAUD"

                            # Fraud banner at bottom of frame.
                            alert_rect_h = 45
                            cv2.rectangle(
                                frame,
                                (0, 480 - alert_rect_h), (640, 480),
                                (0, 0, 200), -1
                            )
                            # Show only the violation type, not the full reason
                            # string (keeps the banner readable on screen).
                            alert_label = reason.split(':')[0]
                            cv2.putText(
                                frame, f"⚠ {alert_label}", (10, 480 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 255, 255), 2
                            )

                    # Draw bounding box and plate label.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{p_text} [{status_msg}]" if status_msg else p_text
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
                    )

                processed_frames.append(frame)

            # ── Compose Dual-Feed Display ────────────────────────────────────
            if len(processed_frames) == 2:
                combined = np.hstack((processed_frames[0], processed_frames[1]))

                # Title header bar.
                header = np.full((65, 1280, 3), 30, dtype=np.uint8)  # dark grey
                cv2.putText(
                    header,
                    "SPATIAL-TEMPORAL VERIFICATION FRAMEWORK  |  RESEARCH DEMO",
                    (310, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

                final_display = np.vstack((header, combined))
                cv2.imshow('Research Viva Demo', final_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("\n🛑 Demo session ended cleanly.")


if __name__ == "__main__":
    run_toy_car_demo()
