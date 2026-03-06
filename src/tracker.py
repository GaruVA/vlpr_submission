import numpy as np
import re

class PlateTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.saved_to_db = set()
        
    def update(self, detections):
        self.frame_count += 1
        updated_tracks = {}
        
        for track_id, track in self.tracks.items():
            best_match_idx, best_iou = -1, 0
            for i, det in enumerate(detections):
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                det = detections[best_match_idx]
                track['bbox'] = det['bbox']
                track['text_history'].append(det['text'])
                track['confidence_history'].append(det['confidence'])
                track['last_seen'] = self.frame_count
                track['hits'] += 1
                track['crop'] = det['crop']
                track['consensus_text'] = self.get_consensus_text(track['text_history'])
                track['avg_confidence'] = np.mean(track['confidence_history'])
                updated_tracks[track_id] = track
                detections.pop(best_match_idx)
            elif self.frame_count - track['last_seen'] < self.max_age:
                updated_tracks[track_id] = track
        
        for det in detections:
            new_track = {
                'bbox': det['bbox'],
                'text_history': [det['text']],
                'confidence_history': [det['confidence']],
                'consensus_text': det['text'],
                'avg_confidence': det['confidence'],
                'hits': 1,
                'last_seen': self.frame_count,
                'first_seen': self.frame_count,
                'crop': det['crop'],
                'saved_to_db': False
            }
            updated_tracks[self.next_id] = new_track
            self.next_id += 1
        
        self.tracks = updated_tracks
        return self.tracks
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def get_consensus_text(self, text_history):
        if not text_history:
            return ""
        
        recent_texts = text_history[-10:]
        text_counts = {}
        for text in recent_texts:
            if text and text != "No text detected":
                text_counts[text] = text_counts.get(text, 0) + 1
        
        if not text_counts:
            return recent_texts[-1] if recent_texts else ""
        
        consensus = max(text_counts.items(), key=lambda x: x[1])[0]
        
        formatted_plates = []
        for text, count in text_counts.items():
            if re.match(r'^[A-Z]{2,3}-\d{1,4}[A-Z]?$', text):
                formatted_plates.append((text, count))
        
        if formatted_plates:
            return max(formatted_plates, key=lambda x: x[1])[0]
        
        return consensus

    def should_save_to_database(self, track):
        return not track.get('saved_to_db', False)
    
    def calculate_plate_similarity(self, plate1, plate2):
        if not plate1 or not plate2:
            return 0.0
        
        if plate1 == plate2:
            return 1.0
        
        confusions = {
            'K': 'X', 'X': 'K', 'D': 'O', 'O': 'D', 
            'B': '8', '8': 'B', 'P': 'B', 'G': 'C', 'C': 'G'
        }
        
        variations = [plate1]
        for i, char in enumerate(plate1):
            if char in confusions:
                variation = plate1[:i] + confusions[char] + plate1[i+1:]
                variations.append(variation)
        
        if plate2 in variations:
            return 0.9
        
        if len(plate1) == len(plate2):
            matches = sum(1 for a, b in zip(plate1, plate2) if a == b)
            return matches / len(plate1)
        
        return 0.0

    def should_save_to_database_check(self, track_id, track, min_hits=3, min_confidence=0.7):
        plate_key = f"{track_id}_{track['consensus_text']}"
        if plate_key in self.saved_to_db:
            return False
        
        if track['hits'] < min_hits:
            return False
            
        if not track.get('is_valid', False):
            return False
        
        if track['avg_confidence'] < min_confidence:
            return False
        
        self.saved_to_db.add(plate_key)
        return True
