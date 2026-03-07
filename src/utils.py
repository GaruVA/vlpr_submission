"""
src/utils.py
Shared Vision Utility Functions

Sprint 3 Addition:
  - enhance_plate_contrast(): CLAHE-based contrast enhancement.

    This function is described in Interim Report Chapter 07, Snippet 2 and
    §7.4 (Challenges and Solutions, Challenge 1):

    "A preprocessing step using OpenCV's CLAHE algorithm was introduced into
    the pipeline. CLAHE divides the image into small local tiles (8×8 pixels
    in this implementation) and equalises contrast within each tile
    independently. This restores legible contrast in the plate region without
    amplifying the background haze, which global histogram equalisation would
    do. Accuracy under hazy conditions improved measurably after this change."

    The function was previously defined (as Snippet 2 in the report) but was
    never called anywhere in the pipeline — a gap between the report's claims
    and the implementation. Sprint 3 wires it into both main_system.py and
    research_demo.py at the correct pipeline stage: AFTER plate crop but
    BEFORE the character recognition model runs.

    Why CLAHE at OCR stage only (not plate detection stage):
      YOLOv8 plate detection operates on the full frame and its internal
      preprocessing handles global contrast adequately. The character
      recognition model, however, receives only the small cropped plate
      region — often just 80×30 pixels — where salt spray deposits create
      localised haze that disproportionately degrades character legibility.
      CLAHE's tile-based local equalisation is specifically designed for
      this scenario.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# enhance_plate_contrast — Interim Report Chapter 07, Snippet 2
# ─────────────────────────────────────────────────────────────────────────────

def enhance_plate_contrast(cropped_plate_img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to a
    cropped plate image to improve character legibility under maritime
    degradation conditions (salt spray haze, sun glare, monsoon rain).

    Directly implements Interim Report Chapter 07, Snippet 2:

        lab = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    Why LAB colour space (not direct BGR/HSV):
      CLAHE should only equalise luminance, not colour channels. Applying
      it to R, G, B independently would create colour-cast artefacts and
      distort the plate's character-to-background contrast ratio.
      The LAB colour model separates luminance (L) from chrominance (A, B),
      so CLAHE on L alone produces a clean, colour-neutral enhancement.

    Why clipLimit=2.0:
      CLAHE limits contrast amplification per tile to 2× the average
      histogram bin height. This prevents noise amplification in uniform
      regions (e.g. a haze-filled plate corner) while still boosting
      contrast in character stroke edges.

    Why tileGridSize=(8, 8):
      8×8 tiles balance local adaptation granularity against boundary
      artefacts. Coarser grids (e.g. 2×2) behave like global equalisation;
      finer grids (e.g. 16×16) produce tiling artefacts at character edges.

    Args:
        cropped_plate_img: BGR uint8 numpy array of the plate region crop.
                           Must be non-empty (caller must validate size > 0).

    Returns:
        BGR uint8 numpy array with CLAHE-enhanced luminance channel.
        Same shape and dtype as input. If enhancement fails for any reason
        (e.g. unexpected colour space), the original image is returned
        unchanged so the pipeline degrades gracefully.
    """
    # Guard: if the crop is grayscale (2D), convert to 3-channel first.
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return cropped_plate_img

    if cropped_plate_img.ndim == 2:
        # Grayscale input — CLAHE can be applied directly to the single channel.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(cropped_plate_img)

    try:
        # ── Interim Report Snippet 2 — verbatim algorithm ────────────────────

        # Step 1: Convert BGR → LAB to isolate the luminance channel.
        lab = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Step 2: Apply CLAHE to the L (luminance) channel only.
        # clipLimit=2.0 and tileGridSize=(8,8) as specified in the report.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Step 3: Reconstruct LAB image with enhanced luminance.
        merged = cv2.merge((cl, a, b))

        # Step 4: Convert back to BGR for downstream processing.
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # ── End Interim Report Snippet 2 ─────────────────────────────────────

    except cv2.error:
        # Colour space conversion failed (e.g. single-pixel crop, unusual
        # bit depth). Return input unchanged — pipeline must not crash on
        # a bad crop during a live Viva demonstration.
        return cropped_plate_img


# ─────────────────────────────────────────────────────────────────────────────
# smart_character_ordering — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def smart_character_ordering(char_boxes: list, plate_shape: tuple) -> tuple:
    """
    Sort detected character bounding boxes into the correct reading order.

    Handles both single-row (standard) and two-row (provincial code prefix)
    Sri Lankan plate layouts. For two-row plates, orders the top row
    (letter prefix) before the bottom row (numeric series).

    Args:
        char_boxes: List of character detection dicts, each with keys:
                    'x', 'y', 'w', 'h', 'char', 'conf', 'is_letter'.
        plate_shape: (height, width, channels) tuple from plate_crop.shape.

    Returns:
        (chars, confidences): Two lists of equal length — the ordered
        character strings and their corresponding confidence scores.
    """
    if not char_boxes:
        return [], []

    plate_height = plate_shape[0]

    # Compute the median Y-centre of all detected characters.
    # This serves as the row-split threshold for two-row plate detection.
    y_positions = sorted(box['y'] + box['h'] / 2 for box in char_boxes)
    median_y = y_positions[len(y_positions) // 2]

    top_row    = [box for box in char_boxes if (box['y'] + box['h'] / 2) < median_y]
    bottom_row = [box for box in char_boxes if (box['y'] + box['h'] / 2) >= median_y]

    # Two-row layout heuristic: at least 2 characters in each row and
    # at least 5 characters total.
    is_two_row = (
        len(top_row) >= 2
        and len(bottom_row) >= 2
        and len(top_row) + len(bottom_row) >= 5
    )

    if is_two_row:
        top_row.sort(key=lambda x: x['x'])
        bottom_row.sort(key=lambda x: x['x'])

        # If the top row is predominantly letters and the bottom row
        # predominantly digits, this is a provincial-prefix plate.
        top_letters    = sum(1 for box in top_row    if box['is_letter'])
        bottom_numbers = sum(1 for box in bottom_row if not box['is_letter'])

        if (top_letters    >= len(top_row)    * 0.6
                and bottom_numbers >= len(bottom_row) * 0.6):
            ordered_boxes = top_row + bottom_row
        else:
            # Mixed layout — fall back to left-to-right single row.
            ordered_boxes = sorted(char_boxes, key=lambda x: x['x'])
    else:
        ordered_boxes = sorted(char_boxes, key=lambda x: x['x'])

    chars       = [box['char'] for box in ordered_boxes]
    confidences = [box['conf'] for box in ordered_boxes]

    return chars, confidences
