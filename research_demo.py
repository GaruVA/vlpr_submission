import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Import refactored modules
from src.tracker import PlateTracker
from src.validator import SriLankanPlateValidator
from src.spatial import SpatialVerifier

# Configuration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

def run_toy_car_demo():
    """
    Simplified demo for 1:18 scale toy cars with DUAL CAMERAS
    Camera 1 (Index 1) = GATE A
    Camera 2 (Index 2) = GATE B
    """
    # Initialize Dual Cameras (Using user's tested indices 1 and 2)
    cam_indices = [1, 2] 
    caps = []
    
    print("🎥 Initializing Dual Camera System...")
    for idx in cam_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"⚠️ Warning: Camera {idx} failed. Trying index 0/fallback...")
            cap = cv2.VideoCapture(0) 
        
        # Set resolution for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps.append(cap)
    
    gate_names = ['GATE A', 'GATE B'] # Corresponding to cam 1, cam 2
    
    # Load your trained models
    plate_model_path = os.path.join(MODELS_DIR, 'plate_detection.pt')
    char_model_path = os.path.join(MODELS_DIR, 'character_recognition.pt')
    
    if not os.path.exists(plate_model_path) or not os.path.exists(char_model_path):
        print(f"Models not found in {MODELS_DIR}")
        return

    print("🧠 Loading YOLOv8 Models...")
    plate_model = YOLO(plate_model_path)
    char_model = YOLO(char_model_path)
    
    # Initialize components
    # Using separate trackers for each gate to handle independent detections correctly
    trackers = [
        PlateTracker(min_hits=1, iou_threshold=0.3), # Tracker for Gate A
        PlateTracker(min_hits=1, iou_threshold=0.3)  # Tracker for Gate B
    ]
    
    validator = SriLankanPlateValidator()
    
    # Define "Physics" for Tabletop Research Demo
    # Rule: Travel between A <-> B takes minimum 5 seconds.
    # Violation: < 5s is Speeding/Teleportation (Clone)
    # The dictionary uses tuples as keys: (origin_gate, dest_gate) -> min_seconds
    spatial = SpatialVerifier({
        ('GATE A', 'GATE B'): 5, 
        ('GATE B', 'GATE A'): 5,
    })
    
    print("\n" + "="*60)
    print("🚗 DUAL-CAMERA RESEARCH DEMO INITIALIZED")
    print(f"   Gate A: Camera {cam_indices[0]}")
    print(f"   Gate B: Camera {cam_indices[1]}")
    print("   Physics Constraint: Min Travel Time = 5.0 seconds")
    print("="*60)
    print("Press 'q' to quit")
    
    try:
        while True:
            processed_frames = []
            
            # Process each camera stream
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                
                # Handle camera dropout gracefully
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "NO SIGNAL", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    frame = cv2.resize(frame, (640, 480))
                
                gate_id = gate_names[i]
                current_tracker = trackers[i]
                
                # --- 1. Plate Detection ---
                # Reducing confidence for demo responsiveness (tabletop lighting is tricky)
                results = plate_model.predict(frame, conf=0.4, verbose=False)
                plate_res = results[0]
                
                detections = []
                
                if len(plate_res.boxes) > 0:
                     for j, box in enumerate(plate_res.boxes.xyxy):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        
                        # Add padding
                        h, w = frame.shape[:2]
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(w, x2); y2 = min(h, y2)
                        
                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size == 0: continue

                        # --- 2. Character Recognition (Simplified) ---
                        plate_text = ""
                        try:
                            # Save temp for char model not needed if passing numpy array
                            # Passing numpy array directly to predict is faster/better if supported
                            char_results = char_model.predict(plate_crop, conf=0.4, verbose=False)[0]
                            
                            found_chars = []
                            if char_results.boxes:
                                for cbox in char_results.boxes:
                                    cls_id = int(cbox.cls[0])
                                    x_center = cbox.xywh[0][0] # use x center to sort
                                    
                                    # Use validator's mapping if available, or fallback
                                    char = validator.class_to_char.get(cls_id, '?')
                                    found_chars.append((x_center, char))
                                
                                # Sort characters left-to-right
                                found_chars.sort(key=lambda x: x[0])
                                plate_text = "".join([c[1] for c in found_chars])
                                
                        except Exception as e:
                            pass # OCR failure is expected in noisy demo, skip text
                        
                        if len(plate_text) < 3: 
                            plate_text = "Reading..."

                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'text': plate_text,
                            'confidence': float(plate_res.boxes.conf[j]),
                            'crop': plate_crop,
                            'is_valid': True # Assume visual valid for demo
                        })

                # --- 3. Tracking & Spatial Verification ---
                tracks = current_tracker.update(detections)
                
                # Overlay specific to this camera
                cv2.putText(frame, f"{gate_id}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                for track_id, track in tracks.items():
                    # For demo, lower hit threshold so it feels "instant"
                    if track['hits'] >= 1: 
                        x1, y1, x2, y2 = track['bbox']
                        p_text = track['consensus_text']
                        
                        box_color = (255, 165, 0) # Orange (Scanning)
                        status_msg = ""
                        
                        # Only verify if we have a reasonable text length
                        if len(p_text) > 4:
                            # CALL SPATIAL LOGIC
                            is_valid, reason = spatial.check_entry(p_text, gate_id)
                            
                            if is_valid:
                                box_color = (0, 255, 0) # Green
                                status_msg = "OK"
                            else:
                                box_color = (0, 0, 255) # Red
                                status_msg = "FRAUD"
                                
                                # Show Alert On Screen
                                alert_rect_h = 40
                                cv2.rectangle(frame, (0, 480-alert_rect_h), (640, 480), (0, 0, 255), -1)
                                cv2.putText(frame, reason.split(':')[0], (10, 480-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{p_text} [{status_msg}]" if status_msg else p_text
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                processed_frames.append(frame)

            # --- 4. Combine Dual Feed ---
            if len(processed_frames) == 2:
                combined = np.hstack((processed_frames[0], processed_frames[1]))
                
                # Create Header
                header_height = 60
                header = np.zeros((header_height, 1280, 3), dtype=np.uint8)
                header[:] = (30, 30, 30) # Dark gray background
                
                cv2.putText(header, "SPATIAL-TEMPORAL VERIFICATION FRAMEWORK (RESEARCH DEMO)", (350, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
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

if __name__ == "__main__":
    run_toy_car_demo()
