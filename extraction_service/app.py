#!/usr/bin/env python3
"""
Killer Sudoku Image Extraction Service

A Flask HTTP service that extracts killer sudoku puzzles from images using
OpenCV for image processing and EasyOCR for text recognition.

Based on the local_extractor.py logic.
"""
import cv2
import numpy as np
import easyocr
import json
import os
import tempfile
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)

# Lazy-loaded EasyOCR reader
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print("Loading EasyOCR model...")
        # Use GPU for faster inference
        _reader = easyocr.Reader(['en'], gpu=True)
        print("EasyOCR model loaded.")
    return _reader


# =============================================================================
# Image Processing Functions
# =============================================================================

def get_warped_grid(image_path):
    """Load image and apply perspective transform to get a 1800x1800 grid."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        g = cnts[0]
        if cv2.contourArea(g) > (img.shape[0] * img.shape[1] * 0.05):
            p = cv2.arcLength(g, True)
            approx = cv2.approxPolyDP(g, 0.02 * p, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
                rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
                M = cv2.getPerspectiveTransform(
                    rect,
                    np.array([[0, 0], [1799, 0], [1799, 1799], [0, 1799]], dtype="float32")
                )
                warped = cv2.warpPerspective(img, M, (1800, 1800))
                return warped
    return cv2.resize(img, (1800, 1800))


def remove_grid_lines(warped, size=9):
    """Remove thick solid 3x3 box boundary lines from the warped image."""
    h, w = warped.shape[:2]
    cell_h, cell_w = h // size, w // size

    if size == 9:
        box_lines = [3, 6]
    else:
        box_lines = [2, 4]

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
    cleaned = gray.copy()
    line_width = 15

    for line_idx in box_lines:
        x = line_idx * cell_w
        x_start = max(0, x - line_width)
        x_end = min(w, x + line_width)
        v_strip = gray[:, x_start:x_end].copy()
        v_kernel = np.ones((25, 1), np.uint8)
        v_strip_inv = 255 - v_strip
        v_opened = cv2.morphologyEx(v_strip_inv, cv2.MORPH_OPEN, v_kernel)
        v_result = cv2.add(v_strip, v_opened)
        cleaned[:, x_start:x_end] = v_result

        y = line_idx * cell_h
        y_start = max(0, y - line_width)
        y_end = min(h, y + line_width)
        h_strip = gray[y_start:y_end, :].copy()
        h_kernel = np.ones((1, 25), np.uint8)
        h_strip_inv = 255 - h_strip
        h_opened = cv2.morphologyEx(h_strip_inv, cv2.MORPH_OPEN, h_kernel)
        h_result = cv2.add(h_strip, h_opened)
        cleaned[y_start:y_end, :] = h_result

    if len(warped.shape) == 3:
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return cleaned


def is_cage_boundary(crop, horizontal=False, near_box_boundary=False):
    """Detect if there's a cage boundary (dashed/dotted line) in the crop."""
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape
    threshold_mult = 0.8 if near_box_boundary else 1.0

    # Method 1: Edge detection
    edges = cv2.Canny(gray, 30, 100)
    mean_val = np.mean(gray)
    dark_mask = (gray < mean_val - 15).astype(np.uint8) * 255
    combined = cv2.bitwise_or(edges, dark_mask)

    if not horizontal:
        profile = np.sum(combined, axis=1) / (w * 255)
    else:
        profile = np.sum(combined, axis=0) / (h * 255)

    is_line = (profile > 0.15).astype(np.uint8)
    transitions = np.abs(np.diff(is_line.astype(int)))
    num_segments = (np.sum(transitions) + (1 if is_line[0] else 0) + (1 if is_line[-1] else 0)) // 2
    coverage = np.mean(is_line)

    if (num_segments >= 4 and coverage >= 0.30) or (coverage >= 0.85 and num_segments >= 3):
        return True

    # Method 1c: High coverage with grayscale variation
    if coverage >= 0.80:
        if not horizontal:
            col_mins = np.min(gray, axis=0)
            darkest_col = np.argmin(col_mins)
            col_start = max(0, darkest_col - 5)
            col_end = min(w, darkest_col + 5)
            strip = gray[:, col_start:col_end]
            line_profile = np.mean(strip, axis=1)
        else:
            row_mins = np.min(gray, axis=1)
            darkest_row = np.argmin(row_mins)
            row_start = max(0, darkest_row - 5)
            row_end = min(h, darkest_row + 5)
            strip = gray[row_start:row_end, :]
            line_profile = np.mean(strip, axis=0)
        if np.std(line_profile) > 15:
            return True

    # Method 1b: Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    edges_enh = cv2.Canny(enhanced, 40, 120)
    mean_enh = np.mean(enhanced)
    dark_mask_enh = (enhanced < mean_enh - 30).astype(np.uint8) * 255
    combined_enh = cv2.bitwise_or(edges_enh, dark_mask_enh)

    if not horizontal:
        profile_enh = np.sum(combined_enh, axis=1) / (w * 255)
    else:
        profile_enh = np.sum(combined_enh, axis=0) / (h * 255)

    is_line_enh = (profile_enh > 0.18).astype(np.uint8)
    transitions_enh = np.abs(np.diff(is_line_enh.astype(int)))
    num_segments_enh = (np.sum(transitions_enh) + (1 if is_line_enh[0] else 0) + (1 if is_line_enh[-1] else 0)) // 2
    coverage_enh = np.mean(is_line_enh)

    if num_segments_enh >= 5 and coverage_enh >= 0.35:
        return True

    # Method 2: Periodic dots detection
    if not horizontal:
        center_x = w // 2
        strip_width = max(6, w // 5)
        strip = gray[:, max(0, center_x - strip_width // 2):min(w, center_x + strip_width // 2)]
        center_profile = np.mean(strip, axis=1)
    else:
        center_y = h // 2
        strip_height = max(6, h // 5)
        strip = gray[max(0, center_y - strip_height // 2):min(h, center_y + strip_height // 2), :]
        center_profile = np.mean(strip, axis=0)

    profile_min = np.min(center_profile)
    profile_max = np.max(center_profile)

    if profile_max - profile_min >= 5:
        threshold = np.mean(center_profile) - 0.3 * (profile_max - profile_min)
        is_dark = (center_profile < threshold).astype(np.uint8)
        kernel = np.ones(3, dtype=np.uint8)
        is_dark_dilated = np.minimum(np.convolve(is_dark, kernel, mode='same'), 1)
        transitions = np.diff(is_dark_dilated.astype(int))
        num_dark_spots = np.sum(transitions == 1)

        if num_dark_spots >= 3:
            dark_positions = np.where(transitions == 1)[0]
            if len(dark_positions) >= 3:
                gaps = np.diff(dark_positions)
                mean_gap = np.mean(gaps)
                gap_std = np.std(gaps)
                if mean_gap > 10 and gap_std < mean_gap * 0.6:
                    return True

    # Method 3: Faint dotted lines - dark pixels in center strip
    trim_pct = 0.25
    if not horizontal:
        y_start = int(h * trim_pct)
        y_end = int(h * (1 - trim_pct))
        center_strip = gray[y_start:y_end, w // 2 - 3:w // 2 + 3]
    else:
        x_start = int(w * trim_pct)
        x_end = int(w * (1 - trim_pct))
        center_strip = gray[h // 2 - 3:h // 2 + 3, x_start:x_end]

    center_min = int(np.min(center_strip))
    center_mean = np.mean(center_strip)

    min_thresh = int(50 * threshold_mult)
    mean_thresh = int(100 * threshold_mult)
    if center_min < min_thresh and center_mean < mean_thresh:
        return True

    dark_pixel_thresh = int(80 * threshold_mult)
    very_dark_count = np.sum(center_strip < dark_pixel_thresh)
    dark_ratio = very_dark_count / center_strip.size
    dark_ratio_thresh = 0.60 if not near_box_boundary else 0.70
    if dark_ratio > dark_ratio_thresh and center_mean < mean_thresh:
        return True

    # Method 4: Blur-based detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if not horizontal:
        blur_center = blurred[y_start:y_end, w // 2 - 3:w // 2 + 3]
    else:
        blur_center = blurred[h // 2 - 3:h // 2 + 3, :]

    blur_min = int(np.min(blur_center))
    blur_mean = np.mean(blur_center)
    blur_min_thresh = int(40 * threshold_mult)
    if blur_min < blur_min_thresh and blur_mean < mean_thresh:
        return True

    # Method 5: Horizontal walls - check right portion
    if horizontal and w > 20 and not near_box_boundary:
        right_portion = blurred[h // 2 - 3:h // 2 + 3, w // 2:]
        right_min = int(np.min(right_portion))
        right_mean = np.mean(right_portion)
        if right_min < 30 and right_mean < 90:
            return True

    # Method 6: FFT-based periodic pattern detection
    if not horizontal:
        center_strip = gray[y_start:y_end, w // 2 - 5:w // 2 + 5]
        profile = np.mean(center_strip, axis=1).astype(np.float32)
    else:
        center_strip = gray[h // 2 - 5:h // 2 + 5, x_start:x_end]
        profile = np.mean(center_strip, axis=0).astype(np.float32)

    if len(profile) > 20:
        profile_norm = profile - np.mean(profile)
        fft = np.abs(np.fft.fft(profile_norm))
        freq_range = fft[2:15]
        max_freq_power = np.max(freq_range)
        mean_freq_power = np.mean(freq_range)

        if mean_freq_power > 0 and max_freq_power / mean_freq_power > 3.0:
            if np.std(profile) > 15:
                return True

    # Method 7: Relaxed blur + enhancement
    blurred_enhanced = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe_blur = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_blur = clahe_blur.apply(blurred_enhanced)

    if not horizontal:
        enh_blur_center = enhanced_blur[y_start:y_end, w // 2 - 3:w // 2 + 3]
    else:
        enh_blur_center = enhanced_blur[h // 2 - 3:h // 2 + 3, :]

    enh_blur_min = int(np.min(enh_blur_center))
    enh_blur_mean = np.mean(enh_blur_center)
    enh_blur_std = np.std(enh_blur_center)

    if enh_blur_min < 30 and enh_blur_mean < 85 and enh_blur_std > 30:
        return True

    # Method 8: Box boundary specific detection
    if near_box_boundary:
        if not horizontal:
            col_mins = np.min(gray, axis=0)
            darkest_col = np.argmin(col_mins)
            col_start = max(0, darkest_col - 8)
            col_end = min(w, darkest_col + 8)
            strip = gray[:, col_start:col_end]
            line_profile = np.mean(strip, axis=1)
        else:
            row_mins = np.min(gray, axis=1)
            darkest_row = np.argmin(row_mins)
            row_start = max(0, darkest_row - 8)
            row_end = min(h, darkest_row + 8)
            strip = gray[row_start:row_end, :]
            line_profile = np.mean(strip, axis=0)

        strip_min = int(np.min(strip))
        strip_mean = np.mean(strip)
        profile_std = np.std(line_profile)
        overall_mean = np.mean(gray)
        darkness_diff = overall_mean - strip_mean

        if strip_min > 35 and strip_min < 60 and darkness_diff > 12 and profile_std > 12:
            return True

        # Method 9: Edge strip pattern detection
        if not horizontal:
            edge_strip = gray[:, 0:15]
            row_mins = np.min(edge_strip, axis=1)
        else:
            edge_strip = gray[0:30, :]
            row_mins = np.min(edge_strip, axis=0)

        threshold = 140
        is_dark = row_mins < threshold
        dark_count = np.sum(is_dark)
        transitions = np.abs(np.diff(is_dark.astype(int)))
        num_segments = (np.sum(transitions) + (1 if is_dark[0] else 0)) // 2

        if num_segments >= 3 and dark_count >= 15:
            dark_positions = np.where(is_dark)[0]
            if len(dark_positions) >= 10:
                gaps = np.diff(dark_positions)
                long_gaps = gaps[gaps > 3]
                if len(long_gaps) >= 2:
                    gap_std = np.std(long_gaps)
                    gap_mean = np.mean(long_gaps)
                    if gap_std < gap_mean * 0.6:
                        return True

    return False


def prepare_for_ocr(crop, scale=2, binary=True):
    """Prepare image crop for OCR."""
    if crop.size == 0:
        return crop
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    if binary:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
        gray = 255 - gray
    return cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT,
                              value=0 if binary else 255)


def ocr_cage_sum(corner_crop, reader_instance, cage_size=None, cell_h=200, cell_w=200):
    """OCR a cage sum from a corner crop using multiple strategies."""
    if corner_crop.size == 0:
        return None, 0

    all_results = []
    crops = [
        (0.35, 0.50, 4),
        (0.32, 0.40, 5),
        (0.32, 0.35, 5),
        (0.28, 0.25, 6),
    ]

    base_left_margin = 8
    if corner_crop.shape[1] > 20:
        left_strip = corner_crop[5:int(cell_h * 0.35), 3:10]
        if left_strip.size > 0:
            gray_strip = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY) if len(left_strip.shape) == 3 else left_strip
            dark_ratio = np.sum(gray_strip < 100) / gray_strip.size if gray_strip.size > 0 else 0
            if dark_ratio > 0.3:
                base_left_margin = 15

    for h_frac, w_frac, scale in crops:
        left_margin = base_left_margin
        right_limit = int(cell_w * w_frac)
        crop = corner_crop[5:int(cell_h * h_frac), left_margin:right_limit]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        ocr_results = reader_instance.readtext(binary, allowlist='0123456789', detail=1)

        for bbox, txt, conf in ocr_results:
            if txt.isdigit():
                val = int(txt)
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                x_ratio = x_center / binary.shape[1]

                if x_ratio > 0.7:
                    continue

                if val >= 100 and txt[0] == '1':
                    corrected_val = int(txt[1:])
                    if 1 <= corrected_val <= 45:
                        all_results.append({
                            'value': corrected_val,
                            'conf': float(conf) * 0.95,
                            'w_frac': w_frac,
                            'x_ratio': x_ratio
                        })
                    continue

                all_results.append({
                    'value': val,
                    'conf': float(conf),
                    'w_frac': w_frac,
                    'x_ratio': x_ratio
                })

    if not all_results:
        return None, 0

    value_results = {}
    for r in all_results:
        v = r['value']
        if v not in value_results:
            value_results[v] = []
        value_results[v].append(r)

    best_candidates = []
    for val, results in value_results.items():
        best_result = max(results, key=lambda x: x['conf'])
        if val >= 10:
            wide_results = [r for r in results if r['w_frac'] >= 0.35]
            if wide_results:
                best_result = max(wide_results, key=lambda x: x['conf'])
        if 1 <= val <= 45:
            best_candidates.append(best_result)

    if not best_candidates:
        best = max(all_results, key=lambda x: x['conf'])
        return best['value'], best['conf']

    single_digit = [r for r in best_candidates if r['value'] < 10]
    two_digit = [r for r in best_candidates if r['value'] >= 10]

    if single_digit and two_digit:
        best_1d = max(single_digit, key=lambda x: x['conf'])
        best_2d = max(two_digit, key=lambda x: x['conf'])
        count_1d = len([r for r in all_results if r['value'] == best_1d['value']])

        wide_2d_results = [r for r in all_results if r['value'] == best_2d['value'] and r['w_frac'] >= 0.35]
        wide_1d_results = [r for r in all_results if r['value'] == best_1d['value'] and r['w_frac'] >= 0.45]
        if len(wide_2d_results) >= 2 and best_2d['conf'] >= 0.85:
            if not (len(wide_1d_results) > 0 and best_1d['conf'] >= best_2d['conf']):
                return best_2d['value'], best_2d['conf']

        if best_2d['value'] // 10 == best_1d['value']:
            wide_crops = [r for r in all_results if r['w_frac'] >= 0.35]
            wide_2d_agree = [r for r in wide_crops if r['value'] == best_2d['value']]
            if len(wide_crops) >= 2 and len(wide_2d_agree) == len(wide_crops) and best_2d['conf'] >= 0.65:
                return best_2d['value'], best_2d['conf']

            if best_1d['conf'] >= 0.95 and best_1d['w_frac'] <= 0.30:
                return best_1d['value'], best_1d['conf']

            if best_2d['w_frac'] >= 0.40 and best_2d['conf'] >= 0.95:
                return best_2d['value'], best_2d['conf']

            if best_1d['conf'] - best_2d['conf'] >= 0.15:
                return best_1d['value'], best_1d['conf']

            narrow_1d = [r for r in all_results if r['value'] == best_1d['value'] and r['w_frac'] <= 0.30]
            if len(narrow_1d) == count_1d and best_2d['conf'] >= 0.85:
                return best_2d['value'], best_2d['conf']

            if best_1d['conf'] >= 0.5:
                return best_1d['value'], best_1d['conf']

        if best_2d['conf'] > best_1d['conf']:
            return best_2d['value'], best_2d['conf']

        return best_1d['value'], best_1d['conf']

    if two_digit:
        for candidate in sorted(two_digit, key=lambda x: -x['conf']):
            val = candidate['value']
            wide_matching = [r for r in all_results if r['value'] == val and r['w_frac'] >= 0.35]
            if len(wide_matching) >= 2:
                return val, candidate['conf']

        best_2d = max(two_digit, key=lambda x: x['conf'])
        if best_2d['conf'] >= 0.5:
            return best_2d['value'], best_2d['conf']

    best = max(best_candidates, key=lambda x: x['conf'])
    return best['value'], best['conf']


def solve_extraction(image_path, size=None, debug=False):
    """Main extraction function - extract killer sudoku from image."""
    if size is None:
        size = 6 if "6x6" in image_path else 9

    reader = get_reader()
    warped = get_warped_grid(image_path)
    if warped is None:
        return None

    cell_h, cell_w = 1800 // size, 1800 // size
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Remove thick grid lines for better cage boundary detection
    warped_clean = remove_grid_lines(warped, size)

    # Box boundary positions
    if size == 9:
        box_cols = [2, 5]
        box_rows = [2, 5]
    else:
        box_cols = [1, 3]
        box_rows = [1, 3]

    # Detect walls
    right_walls = [[False] * size for _ in range(size)]
    bottom_walls = [[False] * size for _ in range(size)]

    for r in range(size):
        for c in range(size - 1):
            is_box_boundary = c in box_cols
            x = (c + 1) * cell_w
            crop = warped_clean[r * cell_h:(r + 1) * cell_h, x - 25:x + 25]
            if is_cage_boundary(crop, horizontal=False, near_box_boundary=is_box_boundary):
                right_walls[r][c] = True
        right_walls[r][size - 1] = True

    for r in range(size - 1):
        for c in range(size):
            is_box_boundary = r in box_rows
            y = (r + 1) * cell_h
            crop = warped_clean[y - 25:y + 25, c * cell_w:(c + 1) * cell_w]
            if is_cage_boundary(crop, horizontal=True, near_box_boundary=is_box_boundary):
                bottom_walls[r][c] = True
    bottom_walls[size - 1] = [True] * size

    # Detect anchor cells (cells with cage sum numbers)
    anchor_cells = set()
    anchor_sums = {}
    for r in range(size):
        for c in range(size):
            y1, x1 = r * cell_h, c * cell_w
            cell_color = warped[y1:y1 + cell_h, x1:x1 + cell_w]
            detected_sum, conf = ocr_cage_sum(cell_color, reader, cell_h=cell_h, cell_w=cell_w)
            if detected_sum is not None and detected_sum > 0 and conf > 0.5:
                anchor_cells.add((r, c))
                anchor_sums[(r, c)] = detected_sum

    # Reconstruct cages using flood fill
    visited = set()
    cages = []
    for r in range(size):
        for c in range(size):
            if (r, c) in visited:
                continue
            cells = []
            q = [(r, c)]
            visited.add((r, c))
            while q:
                cr, cc = q.pop(0)
                cells.append((cr, cc))
                if cc + 1 < size and (cr, cc + 1) not in visited and not right_walls[cr][cc]:
                    visited.add((cr, cc + 1))
                    q.append((cr, cc + 1))
                if cc - 1 >= 0 and (cr, cc - 1) not in visited and not right_walls[cr][cc - 1]:
                    visited.add((cr, cc - 1))
                    q.append((cr, cc - 1))
                if cr + 1 < size and (cr + 1, cc) not in visited and not bottom_walls[cr][cc]:
                    visited.add((cr + 1, cc))
                    q.append((cr + 1, cc))
                if cr - 1 >= 0 and (cr - 1, cc) not in visited and not bottom_walls[cr - 1][cc]:
                    visited.add((cr - 1, cc))
                    q.append((cr - 1, cc))
            cells.sort()
            cages.append({'cells': cells})

    # Post-process: merge cages without anchors
    num_anchors = len(anchor_cells)
    num_cages = len(cages)

    if num_cages > num_anchors * 1.1:
        def merge_orphan_cages():
            nonlocal cages, right_walls, bottom_walls

            cell_to_cage = {}
            for i, cage in enumerate(cages):
                for cell in cage['cells']:
                    cell_to_cage[cell] = i

            orphan_indices = []
            for i, cage in enumerate(cages):
                has_anchor = any(cell in anchor_cells for cell in cage['cells'])
                if not has_anchor:
                    orphan_indices.append(i)

            if not orphan_indices:
                return False

            merged = False
            for orphan_idx in orphan_indices:
                orphan_cage = cages[orphan_idx]
                orphan_cells = set(orphan_cage['cells'])

                adjacent_cages = set()
                for (r, c) in orphan_cells:
                    neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    for nr, nc in neighbors:
                        if 0 <= nr < size and 0 <= nc < size:
                            if (nr, nc) not in orphan_cells:
                                adj_cage_idx = cell_to_cage.get((nr, nc))
                                if adj_cage_idx is not None and adj_cage_idx != orphan_idx:
                                    adj_cage = cages[adj_cage_idx]
                                    if any(cell in anchor_cells for cell in adj_cage['cells']):
                                        adjacent_cages.add((adj_cage_idx, (r, c), (nr, nc)))

                if adjacent_cages:
                    adj_idx, orphan_border, adj_border = next(iter(adjacent_cages))
                    or_, oc = orphan_border
                    ar_, ac = adj_border
                    if or_ == ar_:
                        if oc < ac:
                            right_walls[or_][oc] = False
                        else:
                            right_walls[ar_][ac] = False
                    else:
                        if or_ < ar_:
                            bottom_walls[or_][oc] = False
                        else:
                            bottom_walls[ar_][ac] = False
                    merged = True

            return merged

        for _ in range(50):
            if len(cages) <= num_anchors * 1.1:
                break
            if not merge_orphan_cages():
                break
            # Rebuild cages
            visited = set()
            cages = []
            for r in range(size):
                for c in range(size):
                    if (r, c) in visited:
                        continue
                    cells = []
                    q = [(r, c)]
                    visited.add((r, c))
                    while q:
                        cr, cc = q.pop(0)
                        cells.append((cr, cc))
                        if cc + 1 < size and (cr, cc + 1) not in visited and not right_walls[cr][cc]:
                            visited.add((cr, cc + 1))
                            q.append((cr, cc + 1))
                        if cc - 1 >= 0 and (cr, cc - 1) not in visited and not right_walls[cr][cc - 1]:
                            visited.add((cr, cc - 1))
                            q.append((cr, cc - 1))
                        if cr + 1 < size and (cr + 1, cc) not in visited and not bottom_walls[cr][cc]:
                            visited.add((cr + 1, cc))
                            q.append((cr + 1, cc))
                        if cr - 1 >= 0 and (cr - 1, cc) not in visited and not bottom_walls[cr - 1][cc]:
                            visited.add((cr - 1, cc))
                            q.append((cr - 1, cc))
                    cells.sort()
                    cages.append({'cells': cells})

    # Determine sum cell
    for cage in cages:
        cage_anchors = [cell for cell in cage['cells'] if cell in anchor_cells]
        if cage_anchors:
            cage['sum_cell'] = cage_anchors[0]
            cage['predetected_sum'] = anchor_sums.get(cage_anchors[0])
        else:
            min_r = min(cl[0] for cl in cage['cells'])
            top_cells = [cl for cl in cage['cells'] if cl[0] == min_r]
            cage['sum_cell'] = min(top_cells, key=lambda x: x[1])
            cage['predetected_sum'] = None
    cages.sort(key=lambda x: x['cells'][0])

    # OCR board digits and cage sums
    board = [[0] * size for _ in range(size)]
    cage_sums = {}
    cage_map = [[None] * size for _ in range(size)]

    def get_id(i):
        return chr(ord('a') + i) if i < 26 else 'a' + chr(ord('a') + i - 26)

    sum_cell_to_idx = {c['sum_cell']: i for i, c in enumerate(cages)}

    for r in range(size):
        for c in range(size):
            y1, y2, x1, x2 = r * cell_h, (r + 1) * cell_h, c * cell_w, (c + 1) * cell_w
            cell_img = gray[y1:y2, x1:x2]
            cell_color = warped[y1:y2, x1:x2]

            # Board digit OCR
            m = int(cell_h * 0.25)
            center = cell_img[m:-m, m:-m]
            ocr_in = prepare_for_ocr(center)

            def try_digit_ocr(img):
                results = reader.readtext(img, allowlist='0123456789', detail=1)
                best_val, best_conf = 0, 0
                img_h, img_w = img.shape[:2]
                for bbox, txt, conf in results:
                    if txt.isdigit():
                        v = int(txt)
                        hb = abs(bbox[2][1] - bbox[0][1])
                        wb = abs(bbox[1][0] - bbox[0][0])

                        bbox_cx = (bbox[0][0] + bbox[2][0]) / 2
                        bbox_cy = (bbox[0][1] + bbox[2][1]) / 2
                        norm_cx = bbox_cx / img_w
                        norm_cy = bbox_cy / img_h
                        if norm_cx < 0.30 or norm_cx > 0.70 or norm_cy < 0.30 or norm_cy > 0.70:
                            continue

                        ar = wb / hb if hb > 0 else 0
                        if v == 7 and ar < 0.65:
                            v = 1
                        elif v == 1 and ar > 0.789:
                            v = 7
                        if 1 <= v <= size and conf > best_conf:
                            best_conf = conf
                            best_val = v
                return best_val, best_conf

            val, bc = try_digit_ocr(ocr_in)
            if val == 0 or bc < 0.5:
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(ocr_in, kernel, iterations=3)
                val_d, bc_d = try_digit_ocr(dilated)
                if bc_d >= 0.4 and bc_d > bc:
                    val, bc = val_d, bc_d
            if bc < 0.4:
                val = 0
            board[r][c] = val

            # Cage sum
            if (r, c) in sum_cell_to_idx:
                idx = sum_cell_to_idx[(r, c)]
                cid = get_id(idx)
                predetected = cages[idx].get('predetected_sum')
                if predetected is not None:
                    cage_sums[cid] = predetected
                else:
                    detected_sum, conf = ocr_cage_sum(cell_color, reader, cell_h=cell_h, cell_w=cell_w)
                    if detected_sum is not None:
                        cage_sums[cid] = detected_sum
                    else:
                        cage_sums[cid] = 0

    # Build cage map
    for i, cage in enumerate(cages):
        cid = get_id(i)
        for cr, cc in cage['cells']:
            cage_map[cr][cc] = cid

    # Post-processing: correct sum misreads
    expected_total = 405 if size == 9 else 126
    total_sum = sum(cage_sums.values())
    diff = expected_total - total_sum

    if diff != 0:
        corrections = [
            (1, 11, 10), (2, 12, 10), (3, 13, 10), (4, 14, 10), (5, 15, 10),
            (6, 16, 10), (7, 17, 10), (8, 18, 10), (9, 19, 10),
            (1, 21, 20), (2, 22, 20), (3, 23, 20), (4, 24, 20),
            (1, 31, 30), (2, 32, 30), (3, 33, 30), (4, 34, 30),
            (1, 41, 40), (2, 42, 40), (3, 43, 40), (4, 44, 40), (5, 45, 40),
            (1, 4, 3),
            (1, 7, 6), (7, 1, -6),
        ]

        for cage_id, cage_sum in list(cage_sums.items()):
            for wrong, correct, add in corrections:
                if cage_sum == wrong and correct <= 45:
                    if add == diff or add == diff - (sum(cage_sums.values()) - total_sum):
                        cage_sums[cage_id] = correct
                        diff -= add
                        break
            if diff == 0:
                break

    return {
        "size": size,
        "board": board,
        "cage_map": cage_map,
        "cage_sums": cage_sums,
        "walls": {"right_walls": right_walls, "bottom_walls": bottom_walls}
    }


def validate_extraction(result, size=9):
    """Validate the extracted puzzle structure."""
    errors = []
    expected_total = 405 if size == 9 else 126

    cage_map = result['cage_map']
    cage_sums = result['cage_sums']

    total_sum = sum(cage_sums.values())
    if total_sum != expected_total:
        errors.append(f"Total sum {total_sum} != expected {expected_total}")

    cage_cells = {}
    for r in range(size):
        for c in range(size):
            cid = cage_map[r][c]
            if cid not in cage_cells:
                cage_cells[cid] = []
            cage_cells[cid].append((r, c))

    for cid, cells in cage_cells.items():
        n = len(cells)
        min_sum = n * (n + 1) // 2
        max_sum = n * (2 * size + 1 - n) // 2
        actual_sum = cage_sums.get(cid, 0)

        if actual_sum < min_sum or actual_sum > max_sum:
            errors.append(f"Cage {cid}: sum {actual_sum} invalid for {n} cells (range {min_sum}-{max_sum})")

    return len(errors) == 0, errors


def is_contiguous(cells, size):
    """Check if a set of cells forms a contiguous region."""
    if not cells or len(cells) == 1:
        return True

    cell_set = set(cells)
    visited = set()
    q = [cells[0]]
    visited.add(cells[0])

    while q:
        r, c = q.pop(0)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))

    return len(visited) == len(cells)


def extract_with_improvements(image_path, size=None, debug=False, use_vertex_fallback=True):
    """Main extraction function with all improvements."""
    if size is None:
        size = 6 if "6x6" in image_path else 9

    # Step 1: Initial extraction
    result = solve_extraction(image_path, size=size, debug=debug)
    if result is None:
        print("Initial extraction failed")
        return None

    initial_cages = len(result['cage_sums'])
    initial_sum = sum(result['cage_sums'].values())
    print(f"Initial extraction: {initial_cages} cages, sum={initial_sum}")

    # Step 2: Validate
    is_valid, errors = validate_extraction(result, size)
    if is_valid:
        print("Initial extraction is valid!")
        return result

    print(f"Initial validation failed: {len(errors)} errors")

    # Return best effort result
    print("Returning best effort result")
    return result


# =============================================================================
# Classic Sudoku Extraction
# =============================================================================

def extract_classic_sudoku(image_path, size=9, debug=False):
    """Extract classic sudoku board from image (no cages).

    This handles puzzles with pencil marks by filtering out small text
    and only accepting large, centered digits with strong contrast.
    """
    reader = get_reader()
    warped = get_warped_grid(image_path)
    if warped is None:
        return None

    cell_h, cell_w = 1800 // size, 1800 // size
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    board = [[0] * size for _ in range(size)]

    def has_bold_digit(cell_gray):
        """Check if cell contains a bold (dark) digit vs just pencil marks.

        Bold digits have darker pixels and higher contrast than pencil marks.
        """
        # Get the center region
        h, w = cell_gray.shape
        m = int(min(h, w) * 0.15)
        center = cell_gray[m:-m, m:-m]

        # Check for dark pixels (bold digits are typically < 120)
        # Use a more lenient threshold
        dark_pixels = np.sum(center < 120)
        total_pixels = center.size
        dark_ratio = dark_pixels / total_pixels

        # Also check for very dark pixels (< 60) which indicate bold text
        very_dark = np.sum(center < 60)
        very_dark_ratio = very_dark / total_pixels

        # If we have very dark pixels, it's definitely a bold digit
        if very_dark_ratio > 0.005:
            return True

        # If moderate darkness with decent coverage, also likely bold
        return dark_ratio > 0.03

    def try_digit_ocr(img, min_height_ratio=0.45, min_width_ratio=0.20):
        """Try to OCR a single digit from the image.

        Args:
            img: Input image
            min_height_ratio: Minimum bbox height as ratio of image height
                              to filter out pencil marks (small numbers)
            min_width_ratio: Minimum bbox width as ratio of image width
        """
        results = reader.readtext(img, allowlist='0123456789', detail=1)
        best_val, best_conf = 0, 0
        img_h, img_w = img.shape[:2]

        for bbox, txt, conf in results:
            if txt.isdigit():
                v = int(txt)
                hb = abs(bbox[2][1] - bbox[0][1])
                wb = abs(bbox[1][0] - bbox[0][0])

                # Filter out small pencil marks - require digit to be large
                height_ratio = hb / img_h
                width_ratio = wb / img_w
                if height_ratio < min_height_ratio:
                    continue
                # Width filter (except for digit 1 which is narrow)
                if v != 1 and width_ratio < min_width_ratio:
                    continue

                # Check if bbox is well-centered (strict for classic sudoku)
                bbox_cx = (bbox[0][0] + bbox[2][0]) / 2
                bbox_cy = (bbox[0][1] + bbox[2][1]) / 2
                norm_cx = bbox_cx / img_w
                norm_cy = bbox_cy / img_h
                if norm_cx < 0.30 or norm_cx > 0.70 or norm_cy < 0.30 or norm_cy > 0.70:
                    continue

                # Aspect ratio correction for 1 vs 7
                ar = wb / hb if hb > 0 else 0
                if v == 7 and ar < 0.65:
                    v = 1
                elif v == 1 and ar > 0.789:
                    v = 7

                if 1 <= v <= size and conf > best_conf:
                    best_conf = conf
                    best_val = v
        return best_val, best_conf

    for r in range(size):
        for c in range(size):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell_img = gray[y1:y2, x1:x2]

            # First check if cell has a bold digit (not just pencil marks)
            if not has_bold_digit(cell_img):
                board[r][c] = 0
                continue

            # Focus on center region (avoid borders and pencil marks)
            m = int(cell_h * 0.15)
            center = cell_img[m:-m, m:-m]
            ocr_in = prepare_for_ocr(center)

            val, bc = try_digit_ocr(ocr_in)

            # Try with dilation if low confidence
            if val == 0 or bc < 0.5:
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(ocr_in, kernel, iterations=3)
                val_d, bc_d = try_digit_ocr(dilated)
                if bc_d >= 0.4 and bc_d > bc:
                    val, bc = val_d, bc_d

            if bc < 0.4:
                val = 0

            board[r][c] = val

    return {"board": board}


# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/extract', methods=['POST'])
def extract():
    """
    Extract killer sudoku from uploaded image.

    Request: multipart/form-data with 'image' file
    Response: JSON with board, cage_map, cage_sums
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    size = request.form.get('size', '9')

    try:
        size = int(size)
    except ValueError:
        size = 9

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = extract_with_improvements(
            tmp_path,
            size=size,
            debug=False,
            use_vertex_fallback=False
        )

        if result is None:
            return jsonify({"error": "Failed to extract puzzle from image"}), 500

        response = {
            "board": result.get("board", [[0] * size for _ in range(size)]),
            "cage_map": result.get("cage_map", []),
            "cage_sums": result.get("cage_sums", {}),
        }

        total_sum = sum(response["cage_sums"].values())
        num_cages = len(response["cage_sums"])
        expected_sum = 405 if size == 9 else 126

        response["metadata"] = {
            "total_sum": total_sum,
            "expected_sum": expected_sum,
            "num_cages": num_cages,
            "valid": total_sum == expected_sum
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.route('/extract-cells', methods=['POST'])
def extract_cells():
    """
    Extract specific cell images from a sudoku puzzle image.

    Request: multipart/form-data with 'image' file and 'cells' JSON array of [row, col] pairs
    Response: JSON with cell images as base64
    """
    import base64

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    if 'cells' not in request.form:
        return jsonify({"error": "No cells specified"}), 400

    image_file = request.files['image']
    size = int(request.form.get('size', '9'))

    try:
        cells = json.loads(request.form['cells'])
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid cells JSON"}), 400

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        warped = get_warped_grid(tmp_path)
        if warped is None:
            return jsonify({"error": "Failed to extract grid from image"}), 500

        cell_h, cell_w = 1800 // size, 1800 // size
        cell_images = {}

        for cell in cells:
            row, col = cell[0], cell[1]
            if 0 <= row < size and 0 <= col < size:
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell_img = warped[y1:y2, x1:x2]

                # Encode as base64 PNG
                _, buffer = cv2.imencode('.png', cell_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                cell_images[f"{row},{col}"] = img_base64

        return jsonify({"cells": cell_images})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.route('/extract-classic', methods=['POST'])
def extract_classic():
    """
    Extract classic sudoku from uploaded image.

    Request: multipart/form-data with 'image' file
    Response: JSON with board only
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    size = request.form.get('size', '9')

    try:
        size = int(size)
    except ValueError:
        size = 9

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = extract_classic_sudoku(tmp_path, size=size, debug=False)

        if result is None:
            return jsonify({"error": "Failed to extract puzzle from image"}), 500

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# Model will be loaded lazily on first request to avoid CUDA fork issues

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    print(f"Starting extraction service on {args.host}:{args.port}")
    print("Extraction service ready!")

    app.run(host=args.host, port=args.port, debug=False)
