import os
import time
import sys
import numpy as np
import cv2
import threading
import queue
from datetime import datetime
from ultralytics import YOLO

# Import refactored modules
from src.tracker import PlateTracker
from src.validator import SriLankanPlateValidator
from src.database import DatabaseManager
from src.spatial import SpatialVerifier
from src.utils import smart_character_ordering

# Configuration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

CONFIG = {
    'PLATE_MODEL_PATH': os.path.join(MODELS_DIR, 'plate_detection.pt'),
    'CHAR_MODEL_PATH': os.path.join(MODELS_DIR, 'character_recognition.pt'),
    'VIDEO_SOURCE': "rtsp://admin:Admin%4001%21@192.168.100.132:554/Streaming/Channels/101",
    'CONFIDENCE_THRESHOLD': 0.5,
    'TRACKER_MAX_AGE': 30,
    'TRACKER_MIN_HITS': 3,
    'TRACKER_IOU_THRESHOLD': 0.3,
    'FRAME_SKIP': 3,
    'SAVE_DETECTIONS': True,
    'SHOW_INDIVIDUAL_PLATES': True,
    'DATABASE_ENABLED': True,
    'API_BASE_URL': "https://esystems.cdl.lk/backend-Test/NPRCamera/RFID",
    'DEVICE': '01',
    'IN_OUT': 'I',
    'DB_ASYNC_MODE': True,
    'MIN_CONFIDENCE_FOR_DB': 0.75,
    'MIN_HITS_FOR_DB': 40,
    'MODE': 'PRODUCTION',
    'GATE_ID': 'GATE1',
    'ENABLE_SPATIAL_VERIFICATION': True,
    'SPATIAL_TRAVEL_TIMES': {
        ('GATE1', 'GATE2'): 120,
        ('GATE1', 'GATE3'): 300,
        ('GATE2', 'GATE3'): 180,
    },
}

def run_enhanced_plate_detection():
    """Enhanced plate detection with tracking and consensus"""
    
    if not os.path.exists(CONFIG['PLATE_MODEL_PATH']) or not os.path.exists(CONFIG['CHAR_MODEL_PATH']):
        print("❌ Model paths are invalid")
        print(f"Plate model: {CONFIG['PLATE_MODEL_PATH']}")
        print(f"Char model: {CONFIG['CHAR_MODEL_PATH']}")
        return

    source = CONFIG['VIDEO_SOURCE']
    print("🚗 Enhanced Vehicle License Plate Detection System")
    print("=" * 60)
    print(f"📹 Video Source: {source}")
    print(f"🎯 Plate Model: {os.path.basename(CONFIG['PLATE_MODEL_PATH'])}")
    print(f"🔤 Char Model: {os.path.basename(CONFIG['CHAR_MODEL_PATH'])}")
    if CONFIG.get('DATABASE_ENABLED'):
        print(f"🗄️  Database: ENABLED | Camera: {CONFIG['DEVICE']} | Direction: {CONFIG['IN_OUT']}")
    else:
        print(f"🗄️  Database: DISABLED")
    print("=" * 60)

    try:
        plate_model = YOLO(CONFIG['PLATE_MODEL_PATH'])
        char_model = YOLO(CONFIG['CHAR_MODEL_PATH'])
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    db_manager = None
    saved_plates = set()
    
    # Setup output directories
    detects_dir = os.path.join(SCRIPT_DIR, "detects")
    for d in ["detects", "detects/crops", "detects/raw", "detects/annotated"]:
        p = os.path.join(SCRIPT_DIR, d)
        if not os.path.exists(p):
            os.makedirs(p)

    if CONFIG.get('DATABASE_ENABLED', False):
        try:
            db_manager = DatabaseManager(
                base_url=CONFIG['API_BASE_URL'],
                cam_code='',
                device=CONFIG['DEVICE'],
                in_out=CONFIG['IN_OUT']
            )
        except Exception as e:
            print(f"⚠️  Database init failed: {e}")
    
    tracker = PlateTracker(
        max_age=CONFIG['TRACKER_MAX_AGE'],
        min_hits=CONFIG['TRACKER_MIN_HITS'],
        iou_threshold=CONFIG['TRACKER_IOU_THRESHOLD']
    )
    validator = SriLankanPlateValidator()
    
    spatial_verifier = None
    if CONFIG.get('ENABLE_SPATIAL_VERIFICATION'):
        spatial_verifier = SpatialVerifier(CONFIG.get('SPATIAL_TRAVEL_TIMES', {}))

    cap = cv2.VideoCapture(source)
    
    if source.startswith("rtsp://") or source.startswith("http://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("📡 Configured for RTSP stream")
    
    frame_count = 0
    start_time = time.time()
    
    cv2.namedWindow("Enhanced License Plate Detection", cv2.WINDOW_NORMAL)
    plate_windows = {}
    
    print("🎮 Controls: 'q' to quit, 's' to save current frame")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        
        # Skip frames for performance if RTSP
        if (source.startswith("rtsp://") or source.startswith("http://")) and frame_count % CONFIG['FRAME_SKIP'] != 0:
            continue

        try:
            plate_results = plate_model.predict(source=frame, imgsz=640, conf=CONFIG['CONFIDENCE_THRESHOLD'], verbose=False)
            plate_res = plate_results[0]
        except Exception as e:
            print(f"⚠️ Plate detection error: {e}")
            continue

        annotated_frame = frame.copy()
        detections = []
        
        if len(plate_res.boxes) > 0:
            for i, box in enumerate(plate_res.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                confidence = float(plate_res.boxes.conf[i])
                
                if confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                    continue
                
                # Expand crop slightly
                plate_height = y2 - y1
                plate_width = x2 - x1
                pad_x = int(plate_width * 0.1)
                pad_y = int(plate_height * 0.1)
                x1_crop = max(0, x1 - pad_x)
                y1_crop = max(0, y1 - pad_y)
                x2_crop = min(frame.shape[1], x2 + pad_x)
                y2_crop = min(frame.shape[0], y2 + pad_y)
                
                plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                    continue

                temp_crop_path = os.path.join(tempfile.gettempdir(), f"temp_plate_{frame_count}_{i}.jpg")
                cv2.imwrite(temp_crop_path, plate_crop)
                
                try:
                    char_results = char_model.predict(temp_crop_path, imgsz=640, conf=0.3, verbose=False)
                    char_res = char_results[0]

                    plate_text = "No text detected"
                    char_confidence = 0.0
                    
                    if len(char_res.boxes) > 0:
                        char_boxes = []
                        for j, cbox in enumerate(char_res.boxes.xyxy):
                            x1c, y1c, x2c, y2c = cbox.cpu().numpy().astype(int)
                            char_conf = float(char_res.boxes.conf[j])
                            cls_id = int(char_res.boxes.cls[j])
                            
                            if char_conf > 0.3:
                                char = validator.class_to_char.get(cls_id, str(cls_id))
                                char_boxes.append({
                                    'x': x1c, 'y': y1c, 'w': x2c-x1c, 'h': y2c-y1c,
                                    'char': char, 'conf': char_conf, 'is_letter': char.isalpha()
                                })
                        
                        chars, confidences = smart_character_ordering(char_boxes, plate_crop.shape)
                        
                        if chars and len(chars) >= 4:
                            plate_text = "".join(chars)
                            char_confidence = np.mean(confidences) if confidences else 0.0
                    
                    validated_text, validated_conf, is_valid = validator.validate_and_correct(plate_text, char_confidence)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': validated_text,
                        'confidence': validated_conf,
                        'is_valid': is_valid,
                        'crop': plate_crop
                    })
                    
                except Exception as e:
                    print(f"⚠️ Character recognition error: {e}")
                finally:
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)
        
        tracks = tracker.update(detections)
        
        # Cleanup closed windows
        current_track_ids = set(tracks.keys())
        windows_to_remove = []
        for window_name in list(plate_windows.keys()):
            if window_name.startswith("Plate "):
                try:
                    track_id = int(window_name.split(" ")[1])
                    if track_id not in current_track_ids:
                        cv2.destroyWindow(window_name)
                        windows_to_remove.append(window_name)
                except (ValueError, IndexError):
                    pass
        for window_name in windows_to_remove:
            del plate_windows[window_name]
        
        # Process Tracks
        for track_id, track in tracks.items():
            if track['hits'] >= tracker.min_hits:
                _, _, track['is_valid'] = validator.validate_and_correct(track['consensus_text'], track['avg_confidence'])
                
                # Database & Spatial Logic
                if (track.get('is_valid', False) and not track.get('saved_to_db', False)
                    and track['avg_confidence'] >= CONFIG['MIN_CONFIDENCE_FOR_DB']
                    and track['hits'] >= CONFIG['MIN_HITS_FOR_DB']):
                    
                    # Similar plate check
                    plate_text = track['consensus_text']
                    should_save = True
                    for saved_plate in saved_plates:
                        similarity = tracker.calculate_plate_similarity(plate_text, saved_plate)
                        if similarity > 0.85:
                            should_save = False
                            print(f"🔄 Skipping similar plate: {plate_text} (similar to saved: {saved_plate})")
                            break
                            
                    if should_save:
                        # Spatial Verification
                        is_allowed = True
                        if spatial_verifier:
                             is_allowed, reason = spatial_verifier.check_entry(plate_text, CONFIG.get('GATE_ID', 'GATE1'))
                             if not is_allowed:
                                 print(f"🚨 FRAUD ALERT for {plate_text}: {reason}")
                        
                        # Only save if allowed? Or flag it? 
                        # Usually we save all but mark alerts.
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = f"{track_id}_{plate_text}_{int(track['avg_confidence']*100)}_{timestamp}"
                        
                        # Save images
                        crop_path = os.path.join(detects_dir, "crops", f"crop_{base_name}.jpg")
                        cv2.imwrite(crop_path, track['crop'])
                        
                        saved_plates.add(plate_text)
                        
                        # Database Insert
                        if db_manager and CONFIG.get('DATABASE_ENABLED', False):
                            if CONFIG.get('DB_ASYNC_MODE', True):
                                queued = db_manager.insert_plate_detection(plate_text, track['avg_confidence'], track_id, sync=False)
                                track['saved_to_db'] = bool(queued)
                            else:
                                success = db_manager.insert_plate_detection(plate_text, track['avg_confidence'], track_id, sync=True)
                                track['saved_to_db'] = bool(success)
                    else:
                         track['saved_to_db'] = True # Mark as processed so we don't retry

        # Draw UI
        valid_tracks = 0
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track['bbox']
            if track['hits'] >= tracker.min_hits:
                color = (0, 255, 0) if track.get('is_valid', False) else (0, 165, 255)
                valid_tracks += 1
            else:
                color = (128, 128, 128)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            info_text = f"ID:{track_id} {track['consensus_text']} ({track['avg_confidence']:.2f})"
            cv2.putText(annotated_frame, info_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if (track['hits'] >= tracker.min_hits and CONFIG['SHOW_INDIVIDUAL_PLATES'] and 'crop' in track):
                window_name = f"Plate {track_id}"
                if window_name not in plate_windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    plate_windows[window_name] = True
                cv2.imshow(window_name, cv2.resize(track['crop'], (300, 100)))

        cv2.imshow("Enhanced License Plate Detection", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if db_manager:
        db_manager.shutdown()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG['VIDEO_SOURCE'] = sys.argv[1]
    run_enhanced_plate_detection()
