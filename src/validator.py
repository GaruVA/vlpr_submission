"""
src/validator.py
Layer 2 — Weighted Homoglyph Matching & Plate Format Validation

This module fulfils two academic claims from the Interim Report:

  1. LPM-MLED Algorithm (Interim Report §6.4, Algorithm 1):
     "Modified Levenshtein Edit Distance (LPM-MLED): Standard edit distance
     algorithms assign a uniform substitution cost of 1.0 regardless of which
     characters are involved. The LPM-MLED algorithm addresses this by applying
     a penalty matrix in which optically similar character pairs carry a
     discounted cost (0.1), while unrelated pairs retain the full cost (1.0).
     A match is accepted if the total computed distance is 0.5 or less."

  2. Weighted Fuzzy Validation (Interim Report §3.1, FR-03):
     "The system must compare extracted plate text against the registered
     vehicle database using a Modified Levenshtein Edit Distance algorithm
     configured with an optical confusion penalty matrix. Substitutions that
     correspond to known camera blur artefacts must carry a low penalty, while
     unrelated character substitutions must carry a full penalty."

Sprint 2 Additions:
  - OPTICAL_CONFUSION_MATRIX: module-level constant defining all character
    confusion pairs and their weighted penalties.
  - REGISTERED_VEHICLES: mock in-memory vehicle database for demo.
  - LPM_MLED_THRESHOLD: match acceptance threshold (0.5 per Interim Report).
  - SriLankanPlateValidator.lpm_mled(): O(m×n) DP implementation.
  - SriLankanPlateValidator.find_best_match(): database lookup wrapper.

Sprint 1 Retained:
  - QW-2: valid_prefixes is a set for O(1) lookup.
  - is_reasonable_plate_text(): STC engine gate guard.
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# LPM-MLED: Optical Confusion Penalty Matrix
#
# Interim Report §6.4, Algorithm 1:
# "optically similar character pairs carry a discounted cost (0.1),
#  while unrelated pairs retain the full cost (1.0)"
#
# Each entry (A, B): cost represents the substitution penalty when the
# OCR engine reads character A instead of the ground-truth character B
# (or vice versa — the matrix is symmetric).
#
# The matrix is deliberately exhaustive to cover all common ALPR OCR
# confusions documented in the maritime/industrial ALPR literature
# (Anagnostopoulos et al., 2008; Du et al., 2013).
# ─────────────────────────────────────────────────────────────────────────────

OPTICAL_CONFUSION_MATRIX: dict = {
    # ── High visual similarity — 0.1 penalty ─────────────────────────────────
    # These pairs are visually indistinguishable under salt spray, blur,
    # or low contrast. Treating them as near-free substitutions reduces
    # false negatives without increasing false positives.

    # Digit ↔ Letter — the most common ALPR OCR errors
    ('8', 'B'): 0.1,  ('B', '8'): 0.1,   # vertical bars align identically
    ('0', 'O'): 0.1,  ('O', '0'): 0.1,   # circular shapes
    ('0', 'D'): 0.1,  ('D', '0'): 0.1,   # closed curve vs open curve
    ('0', 'Q'): 0.1,  ('Q', '0'): 0.1,   # zero with descender
    ('1', 'I'): 0.1,  ('I', '1'): 0.1,   # single vertical stroke
    ('1', 'L'): 0.1,  ('L', '1'): 0.1,   # vertical stroke + base
    ('5', 'S'): 0.1,  ('S', '5'): 0.1,   # S-curve shapes
    ('2', 'Z'): 0.1,  ('Z', '2'): 0.1,   # diagonal stroke
    ('6', 'G'): 0.1,  ('G', '6'): 0.1,   # closed top + open bottom
    ('4', 'A'): 0.1,  ('A', '4'): 0.1,   # triangular upper structure
    ('7', 'T'): 0.1,  ('T', '7'): 0.1,   # horizontal top stroke
    ('9', 'P'): 0.1,  ('P', '9'): 0.1,   # closed top loop

    # Letter ↔ Letter — shape-based confusions
    ('C', 'G'): 0.1,  ('G', 'C'): 0.1,   # arc + internal stroke
    ('K', 'X'): 0.1,  ('X', 'K'): 0.1,   # diagonal strokes
    ('U', 'V'): 0.1,  ('V', 'U'): 0.1,   # open-bottom shapes
    ('I', 'J'): 0.1,  ('J', 'I'): 0.1,   # vertical + descender

    # ── Medium visual similarity — 0.4 penalty ────────────────────────────────
    # These pairs have structural overlap but are distinguishable under
    # good imaging conditions. The 0.4 penalty allows matching when
    # imaging is degraded, while resisting false positives on clear footage.

    ('D', 'O'): 0.4,  ('O', 'D'): 0.4,   # rounded closed shapes
    ('P', 'B'): 0.4,  ('B', 'P'): 0.4,   # vertical + upper loop
    ('3', 'E'): 0.4,  ('E', '3'): 0.4,   # horizontal strokes
    ('E', 'F'): 0.4,  ('F', 'E'): 0.4,   # horizontal strokes, one missing
    ('M', 'N'): 0.4,  ('N', 'M'): 0.4,   # multiple vertical strokes
    ('H', 'K'): 0.4,  ('K', 'H'): 0.4,   # crossbar position
    ('R', 'P'): 0.4,  ('P', 'R'): 0.4,   # loop + leg
    ('W', 'M'): 0.4,  ('M', 'W'): 0.4,   # inverted structure
}

# Match acceptance threshold — Interim Report §6.4:
# "A match is accepted if the total computed distance is 0.5 or less."
LPM_MLED_THRESHOLD: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Mock Registered Vehicle Database
#
# Interim Report FR-03 / §6.4:
# "The cleaned string is then matched against the employee vehicle database
#  using the Weighted Homoglyph Matching algorithm."
#
# In production: these records come from the PostgreSQL/SQLite employee-vehicle
# table via a database query.
# In the demo: this frozenset acts as the in-memory vehicle registry.
#
# The dataset is intentionally seeded with OCR-confusion-prone plates so that
# the LPM-MLED matching can be demonstrated during the viva. For example:
#   OCR reads 'CA8-1234' → lpm_mled('CA81234', 'CAB1234') = 0.1 → match ✓
#   OCR reads 'WP-S678'  → lpm_mled('WPS678',  'WP5678')  = 0.1 → match ✓
# ─────────────────────────────────────────────────────────────────────────────

REGISTERED_VEHICLES: frozenset = frozenset({
    # Two-letter prefix series
    'WP-1234', 'WP-5678', 'WP-9012', 'WP-3333',
    'SP-1111', 'SP-2222', 'SP-3333', 'SP-4444',
    'NW-4444', 'NW-5555', 'NW-6666',
    'EP-6666', 'EP-7777', 'EP-8888',
    'NC-8888', 'NC-9999', 'NC-1010',
    'SG-1010', 'SG-2020', 'SG-3030',
    # Three-letter prefix series
    'CAB-1234', 'CAB-5678', 'CAB-9012', 'CAB-3456',
    'UVA-1234', 'UVA-5678', 'UVA-9012',
    'SAB-1234', 'SAB-5678', 'SAB-9999',
    'NCP-1234', 'NCP-5678', 'NCP-3456',
    # Plates specifically chosen to test OCR confusion detection:
    #   'CAB-1234' → OCR may produce 'CA8-1234' (B→8) — must still match
    #   'WP-5678'  → OCR may produce 'WP-S678'  (5→S) — must still match
    #   'SP-1111'  → OCR may produce 'SP-IIII'  (1→I) — must still match
})


# ─────────────────────────────────────────────────────────────────────────────
# SriLankanPlateValidator
# ─────────────────────────────────────────────────────────────────────────────

class SriLankanPlateValidator:
    """
    Layer 2 validation and fuzzy matching for Sri Lankan license plates.

    Provides two distinct matching strategies:

      Rule-based positional correction (validate_and_correct):
        Fast, deterministic. Applies zone-aware character corrections
        based on position (prefix zone = letters, suffix zone = digits).
        Used in the real-time pipeline for every frame.

      LPM-MLED fuzzy matching (find_best_match):
        O(m×n) dynamic programming. Checks the OCR output against the
        full registered vehicle database using the optical confusion
        penalty matrix. Used at the point of database commit to confirm
        the plate is a known registered vehicle (or a plausible OCR
        variant of one).
    """

    def __init__(self) -> None:
        # ── OCR Class → Character Mapping ────────────────────────────────────
        self.class_to_char: dict = {
            0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
            6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
            11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F',
            17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L',
            23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
            29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X',
            35: 'Y', 36: 'Z',
        }

        # ── Valid Format Patterns ─────────────────────────────────────────────
        self.patterns: list = [
            r'^[A-Z]{2}-\d{4}$',    # e.g. WP-1234
            r'^[A-Z]{3}-\d{4}$',    # e.g. CAB-1234
        ]
        self.invalid_patterns: list = [
            r'^[A-Z]{1}-.*',
            r'^.*-\d{1,3}$',
            r'^.*-\d{5,}$',
            r'^.*-\d{1,4}[A-Z]+.*',
            r'^[A-Z]{4,}-.*',
            r'^.*-[A-Z]{2,}$',
        ]

        # ── QW-2 (Sprint 1): valid_prefixes as a SET — O(1) lookup ───────────
        # Previously a list of 18,252 strings causing O(18,252) linear scan.
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.valid_prefixes: set = set()
        for a in letters:
            for b in letters:
                self.valid_prefixes.add(a + b)          # 676 two-letter
                for c in letters:
                    self.valid_prefixes.add(a + b + c)  # 17,576 three-letter
        # Total 18,252 entries — O(1) membership test with set. ───────────────

        # ── Positional Correction Tables ─────────────────────────────────────
        # Applied by validate_and_correct() for fast, rule-based fixes.
        # Separate from the LPM-MLED matrix: these are deterministic
        # zone corrections, not probabilistic distance penalties.
        self.letter_corrections: dict = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
            '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P',
        }
        self.number_corrections: dict = {
            'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4',
            'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9',
            'D': '0', 'Q': '0',
        }
        self.letter_confusions: dict = {
            'K': ['X', 'H'], 'X': ['K'], 'H': ['K'],
            'B': ['8', 'P'], 'P': ['B'], '8': ['B'],
            'D': ['O', '0'], 'O': ['D', '0'], '0': ['O', 'D'],
            'C': ['G'], 'G': ['C'], 'F': ['E'], 'E': ['F'],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # LPM-MLED Algorithm — Interim Report §6.4, Algorithm 1
    # ──────────────────────────────────────────────────────────────────────────

    def lpm_mled(self, s1: str, s2: str) -> float:
        """
        Modified Levenshtein Edit Distance with Optical Confusion Penalty Matrix.

        Implements the LPM-MLED algorithm described in Interim Report §6.4.

        Key difference from standard Levenshtein:
          Standard: all substitutions cost 1.0 uniformly.
          LPM-MLED: substitution cost = OPTICAL_CONFUSION_MATRIX.get((a,b), 1.0)
                    Visually similar pairs (e.g. '8'/'B') cost only 0.1.
                    Unrelated pairs retain the full 1.0 penalty.

        This asymmetry is what makes the algorithm "weighted" — it reduces
        false rejections caused by predictable camera noise while maintaining
        resistance to genuinely different plate strings.

        Both insertion and deletion retain the standard cost of 1.0, since
        length mismatches do not have an optical confusion explanation.

        Algorithm: standard two-row DP (Levenshtein, 1966).
        Time complexity:  O(|s1| × |s2|)
        Space complexity: O(max(|s1|, |s2|))  — two-row optimisation

        Args:
            s1: First string (typically the OCR output after normalisation).
            s2: Second string (typically a registered plate from the database).

        Returns:
            float: The weighted edit distance.
                   ≤ LPM_MLED_THRESHOLD (0.5) → accept as a match.
                   >  LPM_MLED_THRESHOLD       → reject.
        """
        # Normalise: strip dashes and spaces, uppercase both strings.
        # Dashes are structural formatting — including them would penalise
        # correctly formatted plates that differ only in dash presence.
        s1 = s1.replace('-', '').replace(' ', '').upper()
        s2 = s2.replace('-', '').replace(' ', '').upper()

        m, n = len(s1), len(s2)

        # Edge cases: empty string → cost equals the other string's length.
        if m == 0:
            return float(n)
        if n == 0:
            return float(m)

        # ── Two-row DP ────────────────────────────────────────────────────────
        # prev_row[j] = lpm_mled(s1[:i-1], s2[:j])
        # curr_row[j] = lpm_mled(s1[:i],   s2[:j])
        #
        # Base case: transforming empty string into s2[:j] costs j insertions.
        prev_row: list = [float(j) for j in range(n + 1)]

        for i in range(1, m + 1):
            # Base case for this row: transforming s1[:i] into empty = i deletions.
            curr_row: list = [float(i)] + [0.0] * n

            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # Exact character match — zero additional cost.
                    curr_row[j] = prev_row[j - 1]
                else:
                    # Look up the optical confusion penalty for this pair.
                    # Defaults to 1.0 (full penalty) for unrelated characters.
                    sub_cost: float = OPTICAL_CONFUSION_MATRIX.get(
                        (s1[i - 1], s2[j - 1]), 1.0
                    )
                    curr_row[j] = min(
                        prev_row[j]      + 1.0,       # deletion  (standard cost)
                        curr_row[j - 1]  + 1.0,       # insertion (standard cost)
                        prev_row[j - 1]  + sub_cost,  # substitution (weighted)
                    )

            prev_row = curr_row

        return prev_row[n]

    def find_best_match(
        self,
        ocr_text: str,
        registered_plates: 'Optional[list]' = None,
    ) -> tuple:
        """
        Find the closest match in the vehicle registry using the LPM-MLED algorithm.

        Sprint 4 Change (backward-compatible):
          registered_plates replaces the hardcoded REGISTERED_VEHICLES frozenset.
          When called from the CV pipeline, pass db_manager.get_registered_plates()
          to use the live SQLite-managed registry from the dashboard CRUD interface.

          When registered_plates=None (all Sprint 1-3 tests), falls back to
          REGISTERED_VEHICLES frozenset. Fully backward compatible.

        Returns:
            (matched_plate, distance): matched_plate is None if distance > threshold.
        """
        if not ocr_text or ocr_text in ("No text detected", "Reading..."):
            return (None, float('inf'))

        # Sprint 4: live DB list if provided, else frozenset fallback.
        plate_source = registered_plates if registered_plates is not None \
                       else REGISTERED_VEHICLES

        best_plate:    Optional[str] = None
        best_distance: float = float('inf')

        for registered_plate in plate_source:
            distance = self.lpm_mled(ocr_text, registered_plate)
            if distance < best_distance:
                best_distance = distance
                best_plate    = registered_plate

        if best_distance <= LPM_MLED_THRESHOLD:
            return (best_plate, best_distance)
        return (None, best_distance)

    # ──────────────────────────────────────────────────────────────────────────
    # Rule-based validation (Sprint 1 retained, unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def validate_and_correct(
        self, plate_text: str, confidence: float
    ) -> tuple:
        """
        Apply positional corrections, reformat to SL standard, and validate.

        This is the fast path used on every frame in the real-time pipeline.
        find_best_match() is the slower, more accurate path used at DB commit.

        Returns:
            (formatted_text, adjusted_confidence, is_valid)
        """
        if not plate_text or plate_text in ("No text detected", "Reading..."):
            return plate_text, confidence, False

        cleaned   = plate_text.strip().upper()
        corrected = self.apply_positional_corrections(cleaned)
        formatted = self.format_sri_lankan_plate(corrected)

        is_valid   = any(re.match(p, formatted) for p in self.patterns)
        is_invalid = any(re.match(p, formatted) for p in self.invalid_patterns)

        # O(1) set lookup — QW-2 fix from Sprint 1.
        prefix_realistic = False
        dash_pos = formatted.find('-')
        if dash_pos > 0:
            prefix = formatted[:dash_pos]
            prefix_realistic = prefix in self.valid_prefixes

        is_valid = is_valid and not is_invalid and prefix_realistic
        adjusted_confidence = confidence * (1.2 if is_valid else 0.8)

        return formatted, min(adjusted_confidence, 1.0), is_valid

    def apply_positional_corrections(self, text: str) -> str:
        """Zone-aware character corrections based on plate position."""
        if not text:
            return text

        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                prefix, suffix = parts

                if len(prefix) >= 3:
                    last_char = prefix[-1]
                    if last_char.isalpha() and last_char in self.number_corrections:
                        corrected_char = self.number_corrections[last_char]
                        prefix = prefix[:-1]
                        suffix = corrected_char + suffix

                corrected_prefix = ''.join(
                    self.letter_corrections.get(c, c) if c.isdigit() else c
                    for c in prefix
                )
                corrected_suffix = ''.join(
                    self.number_corrections.get(c, c) if c.isalpha() else c
                    for c in suffix
                )
                return f"{corrected_prefix}-{corrected_suffix}"

        clean = text.replace(' ', '').upper()
        corrected = []
        for i, char in enumerate(clean):
            if i < 3:
                corrected.append(
                    self.letter_corrections.get(char, char)
                    if char.isdigit() else char
                )
            else:
                if char.isalpha() and i < len(clean) - 1:
                    corrected.append(self.number_corrections.get(char, char))
                else:
                    corrected.append(char)
        return ''.join(corrected)

    def format_sri_lankan_plate(self, text: str) -> str:
        """Reformat a cleaned plate string to canonical 'XX-NNNN' or 'XXX-NNNN'."""
        if not text:
            return text
        clean = text.replace('-', '').replace(' ', '')
        match = re.match(r'^([A-Z]{2,3})(\d{1,4})([A-Z]?)$', clean)
        if match:
            letters, numbers, suffix = match.groups()
            return f"{letters}-{numbers}{suffix}"
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Module-level utility (Sprint 1: BUG-003 guard for the STC engine)
# ─────────────────────────────────────────────────────────────────────────────

def is_reasonable_plate_text(text: str) -> bool:
    """
    Lightweight sanity gate before calling spatial.check_entry().

    Prevents garbage OCR output ("Reading...", single chars, all-same-char
    strings) from entering the STC engine and generating phantom fraud alerts.

    Enforces:
      - 5 ≤ length ≤ 10
      - Must contain both letters and digits
      - No single-character-dominated strings (> 60% one character)
      - Explicit rejection of known sentinel values
    """
    if not text or len(text) < 5 or len(text) > 10:
        return False

    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    if not (has_letters and has_numbers):
        return False

    for char in set(text):
        if text.count(char) > len(text) * 0.6:
            return False

    if text in ("Reading...", "No text detected"):
        return False

    return True
