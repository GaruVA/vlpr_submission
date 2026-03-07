"""
src/tracker.py
Layer 2 — Consensual Plate Tracking

Sprint 1 Fixes Applied:
  - EDGE-005: `text_history` and `confidence_history` are now capped at
    MAX_HISTORY_LEN (30) entries to prevent unbounded memory growth during
    long gate-dwell sessions (e.g. a vehicle stalling at the barrier).
  - EDGE-007: `tracker.update()` no longer mutates the caller's `detections`
    list in-place. It operates on an internal copy, making the method
    side-effect-free from the caller's perspective.
"""

import numpy as np
import re

# Maximum number of historical readings to retain per track.
# At ~30 FPS a vehicle rarely dwells longer than 1 second at the gate,
# so 30 frames is a generous window that covers the consensus period
# without allowing memory to grow indefinitely.
MAX_HISTORY_LEN: int = 30


class PlateTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: dict = {}
        self.next_id: int = 1
        self.frame_count: int = 0
        self.saved_to_db: set = set()

    def update(self, detections: list) -> dict:
        """
        Match incoming detections to existing tracks via IoU, create new
        tracks for unmatched detections, and age out stale tracks.

        IMPORTANT: This method operates on an internal copy of `detections`.
        The original list passed by the caller is NOT modified. (EDGE-007 fix)

        Args:
            detections: List of detection dicts, each containing keys:
                        'bbox', 'text', 'confidence', 'crop', 'is_valid'.

        Returns:
            Dict mapping track_id (int) -> track state dict.
        """
        self.frame_count += 1
        updated_tracks: dict = {}

        # ── EDGE-007 FIX ────────────────────────────────────────────────────
        # Work on a shallow copy so the caller's list is never mutated.
        unmatched_detections = list(detections)
        # ────────────────────────────────────────────────────────────────────

        for track_id, track in self.tracks.items():
            best_match_idx: int = -1
            best_iou: float = 0.0

            for i, det in enumerate(unmatched_detections):
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i

            if best_match_idx >= 0:
                det = unmatched_detections[best_match_idx]
                track['bbox'] = det['bbox']
                track['text_history'].append(det['text'])
                track['confidence_history'].append(det['confidence'])
                track['last_seen'] = self.frame_count
                track['hits'] += 1
                track['crop'] = det['crop']
                track['consensus_text'] = self.get_consensus_text(track['text_history'])
                track['avg_confidence'] = float(np.mean(track['confidence_history']))

                # ── EDGE-005 FIX ─────────────────────────────────────────────
                # Cap history lists to prevent unbounded memory growth.
                # We keep only the most recent MAX_HISTORY_LEN readings;
                # consensus is computed on this sliding window anyway.
                if len(track['text_history']) > MAX_HISTORY_LEN:
                    track['text_history'] = track['text_history'][-MAX_HISTORY_LEN:]
                    track['confidence_history'] = track['confidence_history'][-MAX_HISTORY_LEN:]
                # ─────────────────────────────────────────────────────────────

                updated_tracks[track_id] = track
                # Remove matched detection so it cannot match another track.
                unmatched_detections.pop(best_match_idx)

            elif self.frame_count - track['last_seen'] < self.max_age:
                # Track has not been seen this frame but hasn't aged out yet.
                updated_tracks[track_id] = track

        # Create new tracks for any detections that went unmatched above.
        for det in unmatched_detections:
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
                'saved_to_db': False,
            }
            updated_tracks[self.next_id] = new_track
            self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute Intersection-over-Union for two (x1,y1,x2,y2) bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left   = max(x1_1, x1_2)
        y_top    = max(y1_1, y1_2)
        x_right  = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Consensus logic
    # ──────────────────────────────────────────────────────────────────────────

    def get_consensus_text(self, text_history: list) -> str:
        """
        Return the most frequently observed plate string from the recent
        history window, preferring correctly formatted Sri Lankan plates.
        """
        if not text_history:
            return ""

        # Only consider the last MAX_HISTORY_LEN readings for consensus.
        recent_texts = text_history[-MAX_HISTORY_LEN:]
        text_counts: dict = {}

        for text in recent_texts:
            if text and text not in ("No text detected", "Reading..."):
                text_counts[text] = text_counts.get(text, 0) + 1

        if not text_counts:
            return recent_texts[-1] if recent_texts else ""

        # Prefer strings that match the canonical SL plate format.
        formatted_plates = [
            (text, count)
            for text, count in text_counts.items()
            if re.match(r'^[A-Z]{2,3}-\d{1,4}[A-Z]?$', text)
        ]

        if formatted_plates:
            return max(formatted_plates, key=lambda x: x[1])[0]

        return max(text_counts.items(), key=lambda x: x[1])[0]

    # ──────────────────────────────────────────────────────────────────────────
    # Database helpers
    # ──────────────────────────────────────────────────────────────────────────

    def should_save_to_database(self, track: dict) -> bool:
        return not track.get('saved_to_db', False)

    def calculate_plate_similarity(self, plate1: str, plate2: str) -> float:
        """
        Heuristic similarity check for de-duplication.
        Used to avoid logging the same physical vehicle twice when the OCR
        flips a single character between reads.
        """
        if not plate1 or not plate2:
            return 0.0
        if plate1 == plate2:
            return 1.0

        confusions = {
            'K': 'X', 'X': 'K', 'D': 'O', 'O': 'D',
            'B': '8', '8': 'B', 'P': 'B', 'G': 'C', 'C': 'G',
        }

        # Check single-character confusion variants.
        variations = [plate1]
        for i, char in enumerate(plate1):
            if char in confusions:
                variation = plate1[:i] + confusions[char] + plate1[i + 1:]
                variations.append(variation)

        if plate2 in variations:
            return 0.9

        if len(plate1) == len(plate2):
            matches = sum(1 for a, b in zip(plate1, plate2) if a == b)
            return matches / len(plate1)

        return 0.0

    def should_save_to_database_check(
        self,
        track_id: int,
        track: dict,
        min_hits: int = 3,
        min_confidence: float = 0.7,
    ) -> bool:
        """Gate check before committing a detection to persistent storage."""
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
