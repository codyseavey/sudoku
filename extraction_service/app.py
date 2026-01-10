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
import pytesseract
import joblib
from pathlib import Path

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


# Lazy-loaded CNN digit classifier
_cnn_classifier = None

def get_cnn_classifier():
    """Get or create the CNN digit classifier.

    Returns None if model file doesn't exist (fallback to EasyOCR only).
    """
    global _cnn_classifier
    if _cnn_classifier is None:
        try:
            from digit_classifier import CNNDigitClassifier, get_model_path
            model_path = get_model_path()
            if os.path.exists(model_path):
                print("Loading CNN digit classifier...")
                _cnn_classifier = CNNDigitClassifier(model_path)
                print("CNN classifier loaded.")
            else:
                print(f"CNN model not found at {model_path}, using EasyOCR only")
                return None
        except Exception as e:
            print(f"Failed to load CNN classifier: {e}, using EasyOCR only")
            return None
    return _cnn_classifier


# Lazy-loaded ML boundary classifier
_boundary_classifier = None
_boundary_scaler = None

def get_boundary_classifier():
    """Get or create the ML boundary classifier.

    Returns (classifier, scaler) tuple if loaded, (None, None) if fallback to heuristics.
    """
    global _boundary_classifier, _boundary_scaler
    if _boundary_classifier is None:
        try:
            model_dir = Path(__file__).parent / 'models'
            classifier_path = model_dir / 'boundary_classifier_rf.pkl'
            scaler_path = model_dir / 'boundary_scaler.pkl'

            if classifier_path.exists() and scaler_path.exists():
                print("Loading ML boundary classifier...")
                _boundary_classifier = joblib.load(classifier_path)
                _boundary_scaler = joblib.load(scaler_path)
                print("ML boundary classifier loaded successfully")
            else:
                print(f"ML boundary model not found at {model_dir}, using heuristic methods")
                return None, None
        except Exception as e:
            print(f"Failed to load ML boundary classifier: {e}, using heuristic methods")
            return None, None
    return _boundary_classifier, _boundary_scaler


# =============================================================================
# OCR Utilities
# =============================================================================

def to_grayscale(img):
    """Convert image to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def add_ocr_border(img, size=20, value=0):
    """Add border for OCR processing."""
    return cv2.copyMakeBorder(img, size, size, size, size,
                               cv2.BORDER_CONSTANT, value=value)


def fix_digit_confusion(digit, bbox_w, bbox_h, conf=1.0):
    """Fix common digit confusions based on aspect ratio.

    OCR often misreads '1' as '7'. Use aspect ratio to correct:
    - '1' is narrow (ar < 0.65) - if OCR says '7' but shape is narrow, correct to '1'
    - '7' is wider - only correct 1→7 if very wide (ar > 0.789)
    """
    if bbox_h <= 0:
        return digit, conf

    ar = bbox_w / bbox_h

    # Narrow shapes labeled '7' are likely '1'
    if digit == 7 and ar < 0.65:
        return 1, conf * 0.95

    # Only very wide shapes labeled '1' might be '7'
    if digit == 1 and ar > 0.789:
        return 7, conf * 0.95

    return digit, conf


class OCRPreprocessor:
    """Unified preprocessing for OCR operations."""

    @staticmethod
    def prepare_digit(crop, scale=2, method='otsu'):
        """Prepare cell crop for digit OCR.

        Args:
            crop: Cell image (grayscale or color)
            scale: Upscaling factor
            method: 'otsu' or 'adaptive'
        """
        if crop.size == 0:
            return crop

        # Convert to grayscale
        gray = to_grayscale(crop)

        # Scale up
        scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scaled)

        # Threshold
        if method == 'otsu':
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(enhanced, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 2)

        # Auto-invert based on corner sampling
        binary = OCRPreprocessor._auto_invert(binary)

        return add_ocr_border(binary)

    @staticmethod
    def prepare_cage_sum(crop, scale=4):
        """Prepare cage sum region for OCR."""
        if crop.size == 0:
            return crop

        gray = to_grayscale(crop)
        scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scaled)

        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return add_ocr_border(binary, value=255)

    @staticmethod
    def _auto_invert(binary):
        """Invert if background is white (corners are bright)."""
        h, w = binary.shape
        corner_size = max(5, h // 10)
        corners = [
            binary[:corner_size, :corner_size],
            binary[:corner_size, -corner_size:],
            binary[-corner_size:, :corner_size],
            binary[-corner_size:, -corner_size:]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])

        if corner_mean > 128:
            return 255 - binary
        return binary


# =============================================================================
# Image Quality Classification
# =============================================================================

class ImageQuality:
    """Image quality classification for preprocessing routing."""
    STANDARD = "standard"       # Normal light background, dark digits
    DARK_MODE = "dark_mode"     # Dark background, light digits
    LOW_CONTRAST = "low_contrast"  # Very light or washed out
    MIXED_LIGHTING = "mixed_lighting"  # Variable intensity across image
    BLURRY = "blurry"           # Low edge sharpness
    UNKNOWN = "unknown"         # Could not classify


def classify_image_quality(img):
    """Classify image quality to route preprocessing.

    Analyzes image characteristics to determine the best preprocessing
    strategy. Returns (quality_type, metrics_dict).

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        tuple: (ImageQuality constant, dict of metrics)
    """
    if img is None or img.size == 0:
        return ImageQuality.UNKNOWN, {}

    # Convert to grayscale for analysis
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Compute metrics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Edge sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_sharpness = laplacian.var()

    # Dark pixel ratio (pixels < 80)
    dark_ratio = np.sum(gray < 80) / gray.size

    # Light pixel ratio (pixels > 200)
    light_ratio = np.sum(gray > 200) / gray.size

    metrics = {
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'edge_sharpness': float(edge_sharpness),
        'dark_ratio': float(dark_ratio),
        'light_ratio': float(light_ratio)
    }

    # Classification logic
    # Dark mode: low mean intensity, high dark ratio
    if mean_intensity < 80 and dark_ratio > 0.6:
        return ImageQuality.DARK_MODE, metrics

    # Low contrast: very high mean (washed out) or very low std
    if mean_intensity > 220 or (mean_intensity > 180 and std_intensity < 30):
        return ImageQuality.LOW_CONTRAST, metrics

    # Mixed lighting: high standard deviation indicates variable illumination
    if std_intensity > 70:
        return ImageQuality.MIXED_LIGHTING, metrics

    # Blurry: low edge sharpness
    if edge_sharpness < 100:
        return ImageQuality.BLURRY, metrics

    # Standard: normal conditions
    return ImageQuality.STANDARD, metrics


def get_preprocessing_params(quality):
    """Get preprocessing parameters based on image quality.

    Returns a dict of parameters to use for various preprocessing steps.
    """
    params = {
        'threshold_method': 'otsu',
        'clahe_clip': 3.0,
        'confidence_threshold': 0.35,
        'use_hsv': False,
        'use_percentile_threshold': False,
        'apply_sharpening': False,
        'multi_scale': False
    }

    if quality == ImageQuality.DARK_MODE:
        params['use_percentile_threshold'] = True
        params['use_hsv'] = True
        params['clahe_clip'] = 4.0  # Higher contrast enhancement
        # Keep default confidence threshold - dark mode is handled by fallback

    elif quality == ImageQuality.LOW_CONTRAST:
        params['threshold_method'] = 'adaptive'
        params['clahe_clip'] = 5.0  # Aggressive contrast enhancement
        params['confidence_threshold'] = 0.30  # Slightly lower threshold

    elif quality == ImageQuality.MIXED_LIGHTING:
        params['threshold_method'] = 'adaptive'
        params['multi_scale'] = True

    elif quality == ImageQuality.BLURRY:
        params['apply_sharpening'] = True
        params['clahe_clip'] = 4.0

    return params


def preprocess_hsv_dark_mode(img):
    """Preprocess dark mode images using HSV color space.

    For dark backgrounds with light digits, uses the Value channel
    and percentile-based thresholding instead of OTSU.

    Args:
        img: BGR input image

    Returns:
        Preprocessed grayscale image (digits white on black)
    """
    if len(img.shape) != 3:
        # Already grayscale, use percentile threshold
        gray = img
    else:
        # Convert to HSV and extract Value channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, gray = cv2.split(hsv)

    # Percentile-based thresholding (better for dark backgrounds)
    p15 = np.percentile(gray, 15)   # Background level
    p85 = np.percentile(gray, 85)   # Digit level

    if p85 - p15 > 25:  # Sufficient contrast
        threshold = (p15 + p85) / 2
        binary = (gray > threshold).astype(np.uint8) * 255
    else:
        # Very low contrast - use adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -5)

    # For dark mode, lighter pixels are digits (already white)
    return binary


def preprocess_hsv_highlighter(img):
    """Remove highlighter marks from images using HSV color space.

    Highlighters have high saturation and high value. This function
    detects and removes highlighter marks to improve digit extraction.

    Args:
        img: BGR input image

    Returns:
        Grayscale image with highlighter removed
    """
    if len(img.shape) != 3:
        return img  # Already grayscale

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Highlighter detection: high saturation + high value
    # Yellow/green/pink highlighters typically S > 50, V > 180
    highlighter_mask = (s > 50) & (v > 180)

    # For pixels under highlighter, use the value channel directly
    # but boost it to compensate for the highlighter darkening ink
    gray = v.copy()

    # Where highlighter is detected, apply local contrast enhancement
    if np.any(highlighter_mask):
        # Create CLAHE for local enhancement
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Replace highlighted regions with enhanced version
        gray[highlighter_mask] = enhanced[highlighter_mask]

    return gray


def apply_sharpening(img):
    """Apply sharpening filter for blurry images.

    Args:
        img: Grayscale input image

    Returns:
        Sharpened grayscale image
    """
    # Unsharp mask: original + (original - blurred) * amount
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened


def get_adaptive_confidence_threshold(confidences, base_threshold=0.35):
    """Calculate adaptive confidence threshold based on distribution.

    Uses the distribution of OCR confidences to set a threshold that
    adapts to the image quality. For high-quality images with many
    confident detections, uses a higher threshold. For difficult images,
    uses a lower threshold.

    Args:
        confidences: List of OCR confidence values
        base_threshold: Fallback threshold if not enough data

    Returns:
        float: Adaptive confidence threshold
    """
    if len(confidences) < 10:
        return base_threshold

    # Use 15th percentile as minimum viable confidence
    # This captures most valid digits while filtering noise
    percentile_threshold = np.percentile(confidences, 15)

    # Clamp to reasonable range [0.20, 0.50]
    adaptive = max(0.20, min(0.50, percentile_threshold))

    # If image is very clean (high mean confidence), use stricter threshold
    mean_conf = np.mean(confidences)
    if mean_conf > 0.8:
        adaptive = max(adaptive, 0.40)

    return adaptive


# =============================================================================
# Structured Error Handling
# =============================================================================

class ExtractionError:
    """Structured error codes for extraction failures."""
    SUCCESS = "success"
    IMAGE_LOAD_FAILED = "image_load_failed"
    GRID_DETECTION_FAILED = "grid_detection_failed"
    NO_DIGITS_FOUND = "no_digits_found"
    CAGE_DETECTION_FAILED = "cage_detection_failed"
    CAGE_SUM_INVALID = "cage_sum_invalid"
    VALIDATION_FAILED = "validation_failed"
    UNKNOWN_ERROR = "unknown_error"


class ExtractionResult:
    """Structured result from extraction with diagnostics."""

    def __init__(self, success=True, error_code=None, data=None, diagnostics=None):
        self.success = success
        self.error_code = error_code or ExtractionError.SUCCESS
        self.data = data or {}
        self.diagnostics = diagnostics or {}

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = {
            'success': self.success,
            'error_code': self.error_code,
        }
        result.update(self.data)
        if self.diagnostics:
            result['diagnostics'] = self.diagnostics
        return result


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

    # Fallback: Try Hough line detection for grid boundary
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=img.shape[0]//5, maxLineGap=30)

    if lines is not None and len(lines) >= 4:
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < img.shape[0] // 6:
                continue
            if angle < 15 or angle > 165:
                h_lines.append((y1+y2)//2)
            elif 75 < angle < 105:
                v_lines.append((x1+x2)//2)

        if len(h_lines) >= 2 and len(v_lines) >= 2:
            h_lines.sort()
            v_lines.sort()
            y1_grid, y2_grid = h_lines[0], h_lines[-1]
            x1_grid, x2_grid = v_lines[0], v_lines[-1]

            # Crop and resize the detected grid region
            if y2_grid > y1_grid + 100 and x2_grid > x1_grid + 100:
                grid_region = img[max(0,y1_grid-10):min(img.shape[0],y2_grid+10),
                                 max(0,x1_grid-10):min(img.shape[1],x2_grid+10)]
                return cv2.resize(grid_region, (1800, 1800))

    # Final fallback: naive resize
    return cv2.resize(img, (1800, 1800))


def detect_grid_lines(warped, size=9):
    """Detect actual grid line positions using projection profiles.

    Returns: (x_positions, y_positions) - lists of size+1 positions for cell boundaries
    """
    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped.copy()

    h, w = gray.shape

    # Apply adaptive threshold to highlight grid lines
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Vertical projection (sum along rows) - finds vertical lines
    v_proj = np.sum(binary, axis=0)  # Shape: (w,)

    # Horizontal projection (sum along columns) - finds horizontal lines
    h_proj = np.sum(binary, axis=1)  # Shape: (h,)

    def find_line_positions(projection, total_size, num_lines):
        """Find num_lines+1 grid line positions from projection profile."""
        expected_spacing = total_size / num_lines

        # Smooth the projection
        kernel_size = int(expected_spacing * 0.1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')

        # Find peaks (grid lines)
        positions = [0]  # Start with 0

        for i in range(1, num_lines):
            # Expected position for this line
            expected_pos = int(i * expected_spacing)

            # Search window around expected position
            search_start = max(0, expected_pos - int(expected_spacing * 0.15))
            search_end = min(total_size, expected_pos + int(expected_spacing * 0.15))

            # Find peak in search window
            window = smoothed[search_start:search_end]
            if len(window) > 0:
                peak_offset = np.argmax(window)
                actual_pos = search_start + peak_offset
                positions.append(actual_pos)
            else:
                positions.append(expected_pos)

        positions.append(total_size)  # End with total size
        return positions

    # Find 10 vertical line positions (for 9 columns)
    x_positions = find_line_positions(v_proj, w, size)

    # Find 10 horizontal line positions (for 9 rows)
    y_positions = find_line_positions(h_proj, h, size)

    return x_positions, y_positions


def get_cell_boundaries(warped, size=9):
    """Get cell boundary positions, with validation and fallback.

    Uses detect_grid_lines() to find actual grid line positions, but falls back
    to uniform spacing if the detection appears unreliable (>15% variance).

    Args:
        warped: 1800x1800 warped grid image
        size: Grid size (6 or 9)

    Returns:
        (x_bounds, y_bounds) where each is a list of size+1 positions marking cell edges
    """
    h, w = warped.shape[:2]
    expected_cell_size = w // size

    # Try to detect actual grid lines
    x_bounds, y_bounds = detect_grid_lines(warped, size)

    # Validate detection - check for reasonable cell sizes
    x_sizes = [x_bounds[i+1] - x_bounds[i] for i in range(size)]
    y_sizes = [y_bounds[i+1] - y_bounds[i] for i in range(size)]

    x_variance = (max(x_sizes) - min(x_sizes)) / expected_cell_size
    y_variance = (max(y_sizes) - min(y_sizes)) / expected_cell_size

    # If variance is too high, fall back to uniform spacing
    if x_variance > 0.15 or y_variance > 0.15:
        x_bounds = [i * expected_cell_size for i in range(size + 1)]
        y_bounds = [i * expected_cell_size for i in range(size + 1)]

    return x_bounds, y_bounds


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


def remove_full_grid_lines(warped_clean, size=9, debug=False):
    """Remove grid lines that span the entire grid.

    Detects and removes thin solid lines at cell boundary positions (1,2,4,5,7,8)
    that span >95% of the grid AND have mean pixel value < 80 (darker lines).
    Grid lines are darker and more continuous than cage boundaries.

    Args:
        warped_clean: Image after box boundary removal
        size: Grid size (9 for 9x9)
        debug: Print debug info

    Returns:
        Image with full grid lines removed
    """
    h, w = warped_clean.shape[:2]
    cell_h, cell_w = h // size, w // size

    gray = cv2.cvtColor(warped_clean, cv2.COLOR_BGR2GRAY) if len(warped_clean.shape) == 3 else warped_clean
    result = gray.copy()

    # Positions to check (exclude box boundaries 3, 6)
    if size == 9:
        check_positions = [1, 2, 4, 5, 7, 8]
    else:
        check_positions = [1, 3, 5]

    # For each potential grid line position
    for line_idx in check_positions:
        # Check VERTICAL line at this position
        x = line_idx * cell_w

        # Sample a strip around this position
        strip_width = 10
        x_start = max(0, x - strip_width)
        x_end = min(w, x + strip_width)
        v_strip = gray[:, x_start:x_end]

        # Find the darkest column in this strip
        col_mins = np.min(v_strip, axis=0)
        darkest_col_local = int(np.argmin(col_mins))
        darkest_col_global = x_start + darkest_col_local

        # Check if this line spans the entire height (>95% coverage)
        line_sample = gray[:, max(0, darkest_col_global-1):min(w, darkest_col_global+2)]
        line_profile = np.min(line_sample, axis=1)

        dark_pixels = np.sum(line_profile < 150)
        coverage = dark_pixels / len(line_profile)
        mean_val = np.mean(line_profile)

        # Remove if high coverage AND dark (grid lines are darker than cage boundaries)
        if coverage > 0.95 and mean_val < 80:
            # This is a full grid line - remove it
            if debug:
                print(f"  Removing full vertical grid line at position {line_idx} (coverage={coverage:.2%}, mean={mean_val:.1f})")

            # Remove using morphological opening
            strip_to_clean = result[:, x_start:x_end].copy()
            v_kernel = np.ones((15, 1), np.uint8)
            strip_inv = 255 - strip_to_clean
            opened = cv2.morphologyEx(strip_inv, cv2.MORPH_OPEN, v_kernel)
            result[:, x_start:x_end] = cv2.add(strip_to_clean, opened)

        # Check HORIZONTAL line at this position
        y = line_idx * cell_h

        # Sample a strip around this position
        y_start = max(0, y - strip_width)
        y_end = min(h, y + strip_width)
        h_strip = gray[y_start:y_end, :]

        # Find the darkest row in this strip
        row_mins = np.min(h_strip, axis=1)
        darkest_row_local = int(np.argmin(row_mins))
        darkest_row_global = y_start + darkest_row_local

        # Check if this line spans the entire width (>95% coverage)
        line_sample = gray[max(0, darkest_row_global-1):min(h, darkest_row_global+2), :]
        line_profile = np.min(line_sample, axis=0)

        dark_pixels = np.sum(line_profile < 150)
        coverage = dark_pixels / len(line_profile)
        mean_val = np.mean(line_profile)

        # Remove if high coverage AND dark (grid lines are darker than cage boundaries)
        if coverage > 0.95 and mean_val < 80:
            # This is a full grid line - remove it
            if debug:
                print(f"  Removing full horizontal grid line at position {line_idx} (coverage={coverage:.2%}, mean={mean_val:.1f})")

            # Remove using morphological opening
            strip_to_clean = result[y_start:y_end, :].copy()
            h_kernel = np.ones((1, 15), np.uint8)
            strip_inv = 255 - strip_to_clean
            opened = cv2.morphologyEx(strip_inv, cv2.MORPH_OPEN, h_kernel)
            result[y_start:y_end, :] = cv2.add(strip_to_clean, opened)

    if len(warped_clean.shape) == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def filter_grid_line_false_positives(right_walls, bottom_walls, size=9, debug=False):
    """Remove boundaries that follow grid line patterns.

    Grid lines exist at EVERY cell in a row/column at the same position.
    If >6 out of 9 cells have a boundary at the same position, it's likely a grid line.

    Args:
        right_walls: 2D array (size x size) of vertical boundaries
        bottom_walls: 2D array (size x size) of horizontal boundaries
        size: Grid size (9 for 9x9 sudoku)
        debug: If True, print debug info

    Returns:
        Filtered (right_walls, bottom_walls) with grid line false positives removed
    """
    # Threshold: ONLY filter positions where ALL cells have a boundary
    # This is extremely conservative to avoid removing any legitimate cage boundaries
    # Even if grid lines are present at 8/9 cells, we won't filter them
    # Phase 2 (sum-based correction) will handle remaining false positives
    threshold = size - 1  # For size=9, >8 means exactly 9/9 (100%)

    # Check vertical boundaries (right_walls)
    # For each column position (0 to size-2), count how many rows have a boundary
    for col_idx in range(size - 1):
        boundary_count = sum(1 for r in range(size) if right_walls[r][col_idx])

        # If >threshold cells have a boundary at this column, it's likely a grid line
        if boundary_count > threshold:
            if debug:
                print(f"  Vertical col {col_idx}: {boundary_count}/{size} boundaries (>threshold={threshold}) - REMOVING")
            # Remove all boundaries at this column position
            for r in range(size):
                right_walls[r][col_idx] = False

    # Check horizontal boundaries (bottom_walls)
    # For each row position (0 to size-2), count how many columns have a boundary
    for row_idx in range(size - 1):
        boundary_count = sum(1 for c in range(size) if bottom_walls[row_idx][c])

        # If >threshold cells have a boundary at this row, it's likely a grid line
        if boundary_count > threshold:
            if debug:
                print(f"  Horizontal row {row_idx}: {boundary_count}/{size} boundaries (>threshold={threshold}) - REMOVING")
            # Remove all boundaries at this row position
            for c in range(size):
                bottom_walls[row_idx][c] = False

    return right_walls, bottom_walls


def extract_features_from_crop(crop, horizontal=False):
    """Extract 38-dimensional feature vector from boundary crop for ML classifier.

    Features (38 total):
    - Basic stats (4): mean, std, min, max
    - Edge detection (3): Canny edge count, edge density, edge coverage
    - Gradient (4): Sobel X/Y magnitude mean, std
    - FFT (6): peak frequency power, mean frequency power, ratio, dominant frequency, profile std, range
    - Darkness (8): dark pixel ratio (<80), very dark ratio (<40), center darkness, background brightness
    - Profile analysis (6): line profile std, min, max, coverage ratio, num segments
    - Morphological (4): opening response, closing response, area, perimeter
    - Texture (3): Local Binary Pattern histogram statistics

    Returns:
        feature_vector: np.array of shape (38,)
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape

    features = []

    # Basic stats (4)
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray)
    ])

    # Edge detection (3)
    edges = cv2.Canny(gray, 30, 100)
    edge_count = np.sum(edges > 0)
    edge_density = edge_count / (h * w)
    edge_coverage = np.mean(edges > 0)
    features.extend([edge_count, edge_density, edge_coverage])

    # Gradient magnitude (4)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([
        np.mean(gradient_mag),
        np.std(gradient_mag),
        np.mean(np.abs(sobelx)),
        np.mean(np.abs(sobely))
    ])

    # FFT analysis (6)
    if not horizontal:
        profile = np.mean(gray, axis=1).astype(np.float32)
    else:
        profile = np.mean(gray, axis=0).astype(np.float32)

    if len(profile) > 20:
        profile_norm = profile - np.mean(profile)
        fft = np.abs(np.fft.fft(profile_norm))
        freq_range = fft[2:15]
        max_freq_power = np.max(freq_range)
        mean_freq_power = np.mean(freq_range)
        ratio = max_freq_power / mean_freq_power if mean_freq_power > 0 else 0
        dominant_freq = np.argmax(freq_range) + 2
        features.extend([max_freq_power, mean_freq_power, ratio, dominant_freq, np.std(profile), np.max(profile) - np.min(profile)])
    else:
        features.extend([0, 0, 0, 0, 0, 0])

    # Darkness analysis (8)
    dark_ratio = np.mean(gray < 80)
    very_dark_ratio = np.mean(gray < 40)
    center_strip = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_strip) if center_strip.size > 0 else np.mean(gray)
    center_min = np.min(center_strip) if center_strip.size > 0 else np.min(gray)
    background = np.max(gray)
    features.extend([
        dark_ratio,
        very_dark_ratio,
        center_mean,
        center_min,
        background,
        background - center_mean,  # Contrast
        np.mean(gray < 120),  # Moderate darkness ratio
        np.mean(gray < 60)   # Dark ratio
    ])

    # Profile analysis (6)
    line_profile = np.min(gray, axis=1) if not horizontal else np.min(gray, axis=0)
    profile_std = np.std(line_profile)
    profile_min = np.min(line_profile)
    profile_max = np.max(line_profile)
    threshold = np.mean(line_profile) - 0.3 * (profile_max - profile_min)
    is_dark = line_profile < threshold
    transitions = np.sum(np.abs(np.diff(is_dark.astype(int))))
    num_segments = transitions // 2
    coverage = np.mean(is_dark)
    features.extend([profile_std, profile_min, profile_max, coverage, num_segments, transitions])

    # Morphological operations (4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    opening_response = np.mean(np.abs(gray.astype(np.float32) - opening.astype(np.float32)))
    closing_response = np.mean(np.abs(gray.astype(np.float32) - closing.astype(np.float32)))

    # Simple area/perimeter from binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    features.extend([opening_response, closing_response, total_area, total_perimeter])

    # Texture - Local Binary Pattern (3)
    # Simplified LBP: compare center with 8 neighbors
    lbp_hist = []
    if h > 2 and w > 2:
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp_hist.append(code)

    if lbp_hist:
        lbp_mean = np.mean(lbp_hist)
        lbp_std = np.std(lbp_hist)
        lbp_max = np.max(lbp_hist)
    else:
        lbp_mean = lbp_std = lbp_max = 0
    features.extend([lbp_mean, lbp_std, lbp_max])

    return np.array(features, dtype=np.float32)


def is_cage_boundary_ml(crop, horizontal=False, near_box_boundary=False, threshold_mult=1.0):
    """ML-based boundary detection using trained Random Forest classifier.

    Args:
        crop: Image crop at potential boundary position
        horizontal: True for horizontal boundaries, False for vertical
        near_box_boundary: True if near 3×3 box boundary
        threshold_mult: Confidence threshold multiplier (>1.0 = stricter, <1.0 = looser)

    Returns:
        is_boundary: Boolean indicating if boundary detected
    """
    classifier, scaler = get_boundary_classifier()

    # Fallback to heuristic methods if ML model not loaded
    if classifier is None or scaler is None:
        return is_cage_boundary_heuristic(crop, horizontal, near_box_boundary, threshold_mult)

    if crop.size == 0:
        return False

    # Extract same 38 features used in training
    features = extract_features_from_crop(crop, horizontal)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict with confidence
    proba = classifier.predict_proba(features_scaled)[0, 1]

    # Apply threshold (default 0.4 to reduce false negatives, adjustable via threshold_mult)
    # Lowered from 0.5 to 0.4 to detect more boundaries (Puzzles 7, 21 had missing boundaries)
    base_threshold = 0.4
    adjusted_threshold = base_threshold * threshold_mult

    is_boundary = proba > adjusted_threshold

    return is_boundary


def is_cage_boundary(crop, horizontal=False, near_box_boundary=False, threshold_mult=1.0):
    """Detect if there's a cage boundary (dashed/dotted line) in the crop.

    Uses ML-based detection with automatic fallback to heuristic methods if ML model unavailable.

    Args:
        crop: Image crop at potential boundary position
        horizontal: True for horizontal boundaries, False for vertical
        near_box_boundary: True if near 3×3 box boundary
        threshold_mult: Confidence threshold multiplier (>1.0 = stricter, <1.0 = looser)

    Returns:
        is_boundary: Boolean indicating if boundary detected
    """
    # Use ML classifier (automatically falls back to heuristic if model not loaded)
    return is_cage_boundary_ml(crop, horizontal, near_box_boundary, threshold_mult)


def is_cage_boundary_heuristic(crop, horizontal=False, near_box_boundary=False, threshold_mult=1.0):
    """Heuristic-based boundary detection using ensemble voting (fallback for ML).

    Searches for boundaries ANYWHERE in the crop, not just the center, to be
    robust to grid alignment errors.

    Uses confidence scoring and ensemble voting for robust detection.
    Each method returns a confidence score (0.0-1.0) instead of boolean.

    Args:
        threshold_mult: Multiplier for ensemble voting thresholds (>1.0 = stricter, <1.0 = looser)
    """
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape
    # REMOVED: threshold_mult was making detection more lenient near box boundaries
    # This was causing false positives. Now using consistent thresholds everywhere.

    # Track method confidences (method_name, confidence_score)
    # Confidence ranges: 0.0 = no boundary, 1.0 = very confident boundary
    confidences = []

    # Method 0: Find darkest column/row and check if it's a dashed line
    # This is robust to grid alignment errors since it searches entire crop
    # IMPORTANT: Must be selective to avoid false positives from digits/artifacts
    if not horizontal:
        # For vertical boundaries, find the darkest column
        col_mins = np.min(gray, axis=0)
        darkest_col = int(np.argmin(col_mins))

        # Check if there's a clear dark line in this column
        # Require the line to be within 40% of center to avoid detecting adjacent boundaries
        center_tolerance = 0.40
        if abs(darkest_col - w // 2) > w * center_tolerance:
            pass  # Line too far from center, skip Method 0
        elif col_mins[darkest_col] < 120:  # Relaxed threshold to catch fainter lines
            # Skip very dark solid lines - these are likely residual 3x3 box boundaries
            # Box boundaries are very dark (< 15) and thick
            if col_mins[darkest_col] < 15:
                # Check if this is a thick solid line (likely box boundary)
                # vs an anti-aliased thin line (cage boundary)
                # Anti-aliased lines have a gradient pattern centered on the peak
                very_dark_cols = np.sum(col_mins < 30)

                # Find the largest CONTIGUOUS dark region around the peak using MEANS
                # (col_mins is too sensitive - picks up stray dark pixels)
                col_means = np.mean(gray, axis=0)
                background = np.max(col_means)  # Brightest column is background
                dark_threshold = background * 0.6  # Columns significantly darker than bg
                is_dark = col_means < dark_threshold

                # Find contiguous dark groups and get the largest one
                dark_region_width = 0
                current_run = 0
                for dark in is_dark:
                    if dark:
                        current_run += 1
                        dark_region_width = max(dark_region_width, current_run)
                    else:
                        current_run = 0

                # A thick box boundary would have many uniformly dark columns
                # An anti-aliased thin line has a narrow gradient region (<= 12 cols)
                if very_dark_cols >= 8 and dark_region_width >= 15:
                    pass  # Wide thick line, skip it
                else:
                    # Thin or anti-aliased line - might be a cage boundary
                    col_start = max(0, darkest_col - 2)
                    col_end = min(w, darkest_col + 3)
                    strip = gray[:, col_start:col_end]
                    line_profile = np.min(strip, axis=1)
                    dark_count = np.sum(line_profile < 130)
                    coverage_ratio = dark_count / len(line_profile)

                    # Check contrast: real cage boundary has dark line on light background
                    col_means = np.mean(gray, axis=0)
                    background_brightness = np.max(col_means)

                    # Accept if: good coverage AND high contrast (background > 120)
                    # This accepts most real boundaries
                    if coverage_ratio >= 0.7 and background_brightness > 120:
                        # Confidence based on coverage ratio
                        conf = min(1.0, coverage_ratio)  # 0.7→0.7, 1.0→1.0
                        confidences.append(('method_0_solid', conf))
            else:
                col_start = max(0, darkest_col - 2)
                col_end = min(w, darkest_col + 3)
                strip = gray[:, col_start:col_end]

                # Check for dashed pattern (alternating dark/light along the line)
                line_profile = np.min(strip, axis=1)

                # Use adaptive threshold based on profile statistics
                profile_max = np.max(line_profile)
                profile_min = np.min(line_profile)
                dark_threshold = max(80, profile_min + 0.3 * (profile_max - profile_min))

                # Must have good coverage of dark pixels along the length
                dark_count = np.sum(line_profile < dark_threshold)
                coverage_ratio = dark_count / len(line_profile)

                # Count actual dashes (need distinct on/off pattern)
                is_dark = line_profile < dark_threshold

                # Smooth to avoid noise
                kernel = np.ones(5) / 5
                smoothed = np.convolve(is_dark.astype(float), kernel, mode='same')
                is_dark_smooth = smoothed > 0.5

                transitions = np.abs(np.diff(is_dark_smooth.astype(int)))
                num_dashes = np.sum(transitions) // 2

                # Dashed line: multiple dashes with good coverage
                if num_dashes >= 4 and coverage_ratio >= 0.3:
                    # Confidence based on dash count and coverage
                    dash_conf = min(1.0, num_dashes / 6.0)
                    coverage_conf = min(1.0, coverage_ratio / 0.5)
                    conf = (dash_conf + coverage_conf) / 2
                    confidences.append(('method_0_dashed', max(0.4, conf)))

                # Solid/near-solid line: high coverage with consistent darkness
                # Only accept if background is light (> 150) to avoid grid line residue
                if coverage_ratio >= 0.7 and num_dashes <= 2:
                    col_means = np.mean(gray, axis=0)
                    background_brightness = np.max(col_means)
                    strip_mean = np.mean(strip, axis=1)
                    if np.std(strip_mean) < 45 and background_brightness > 120:
                        # Confidence based on coverage and consistency
                        conf = min(1.0, coverage_ratio)
                        confidences.append(('method_0_solid', conf))
    else:
        # For horizontal boundaries, find the darkest row
        row_mins = np.min(gray, axis=1)
        darkest_row = int(np.argmin(row_mins))

        # Require the line to be within 40% of center to avoid detecting adjacent boundaries
        center_tolerance = 0.40
        if abs(darkest_row - h // 2) > h * center_tolerance:
            pass  # Line too far from center, skip Method 0
        elif row_mins[darkest_row] < 120:  # Relaxed threshold
            # Skip very dark solid lines - likely residual 3x3 box boundaries
            if row_mins[darkest_row] < 15:
                # Check if this is a thick solid line (likely box boundary)
                # vs an anti-aliased thin line (cage boundary)
                very_dark_rows = np.sum(row_mins < 30)

                # Find the largest CONTIGUOUS dark region around the peak using MEANS
                row_means = np.mean(gray, axis=1)
                background = np.max(row_means)
                dark_threshold = background * 0.6
                is_dark = row_means < dark_threshold

                # Find contiguous dark groups and get the largest one
                dark_region_height = 0
                current_run = 0
                for dark in is_dark:
                    if dark:
                        current_run += 1
                        dark_region_height = max(dark_region_height, current_run)
                    else:
                        current_run = 0

                # A thick box boundary would have many uniformly dark rows
                # An anti-aliased thin line has a narrow gradient region
                if very_dark_rows >= 8 and dark_region_height >= 15:
                    pass  # Wide thick line, skip it
                else:
                    # Thin or anti-aliased line - might be a cage boundary
                    row_start = max(0, darkest_row - 2)
                    row_end = min(h, darkest_row + 3)
                    strip = gray[row_start:row_end, :]
                    line_profile = np.min(strip, axis=0)
                    dark_count = np.sum(line_profile < 130)
                    coverage_ratio = dark_count / len(line_profile)

                    # Check contrast: real cage boundary has dark line on light background
                    row_means = np.mean(gray, axis=1)
                    background_brightness = np.max(row_means)

                    # Accept if: good coverage AND high contrast (background > 120)
                    # This accepts most real boundaries
                    if coverage_ratio >= 0.7 and background_brightness > 120:
                        # Confidence based on coverage ratio
                        conf = min(1.0, coverage_ratio)  # 0.7→0.7, 1.0→1.0
                        confidences.append(('method_0_solid', conf))
            else:
                row_start = max(0, darkest_row - 2)
                row_end = min(h, darkest_row + 3)
                strip = gray[row_start:row_end, :]

                line_profile = np.min(strip, axis=0)

                # Use adaptive threshold based on profile statistics
                profile_max = np.max(line_profile)
                profile_min = np.min(line_profile)
                dark_threshold = max(80, profile_min + 0.3 * (profile_max - profile_min))

                dark_count = np.sum(line_profile < dark_threshold)
                coverage_ratio = dark_count / len(line_profile)

                is_dark = line_profile < dark_threshold

                kernel = np.ones(5) / 5
                smoothed = np.convolve(is_dark.astype(float), kernel, mode='same')
                is_dark_smooth = smoothed > 0.5

                transitions = np.abs(np.diff(is_dark_smooth.astype(int)))
                num_dashes = np.sum(transitions) // 2

                # Dashed line: multiple dashes with good coverage
                if num_dashes >= 4 and coverage_ratio >= 0.3:
                    # Confidence based on dash count and coverage
                    dash_conf = min(1.0, num_dashes / 6.0)
                    coverage_conf = min(1.0, coverage_ratio / 0.5)
                    conf = (dash_conf + coverage_conf) / 2
                    confidences.append(('method_0_dashed', max(0.4, conf)))

                # Solid/near-solid line: high coverage with consistent darkness
                # Only accept if background is light (> 150) to avoid grid line residue
                if coverage_ratio >= 0.7 and num_dashes <= 2:
                    row_means = np.mean(gray, axis=1)
                    background_brightness = np.max(row_means)
                    strip_mean = np.mean(strip, axis=0)
                    if np.std(strip_mean) < 45 and background_brightness > 120:
                        return True

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
        # Confidence based on coverage and segment count
        coverage_conf = min(1.0, coverage / 0.85)
        segment_conf = min(1.0, num_segments / 5.0)
        conf = (coverage_conf + segment_conf) / 2
        confidences.append(('method_1', max(0.4, conf)))

    # Method 1c: High coverage - search for darkest column/row without center constraint
    if coverage >= 0.60:
        # Define middle region (avoid top 25% and bottom 25% to skip cage sums)
        mid_start_y = int(h * 0.25)
        mid_end_y = int(h * 0.75)
        mid_start_x = int(w * 0.25)
        mid_end_x = int(w * 0.75)

        if not horizontal:
            mid_gray = gray[mid_start_y:mid_end_y, :]
            col_mins = np.min(mid_gray, axis=0)
            darkest_col = int(np.argmin(col_mins))
            # Accept darkest column anywhere in the crop
            col_start = max(0, darkest_col - 5)
            col_end = min(w, darkest_col + 5)
            strip = mid_gray[:, col_start:col_end]
            line_profile = np.mean(strip, axis=1)
            if np.std(line_profile) > 15 and np.min(strip) < 120:
                # Confidence based on std strength and darkness
                std_conf = min(1.0, np.std(line_profile) / 30)
                dark_conf = (120 - np.min(strip)) / 120
                conf = (std_conf + dark_conf) / 2
                confidences.append(('method_1c', max(0.4, conf)))
        else:
            mid_gray = gray[:, mid_start_x:mid_end_x]
            row_mins = np.min(mid_gray, axis=1)
            darkest_row = int(np.argmin(row_mins))
            row_start = max(0, darkest_row - 5)
            row_end = min(h, darkest_row + 5)
            strip = mid_gray[row_start:row_end, :]
            line_profile = np.mean(strip, axis=0)
            if np.std(line_profile) > 15 and np.min(strip) < 120:
                confidences.append(('method_1c', 1.0))

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

    # More lenient segment requirement for very high coverage
    if num_segments_enh >= 5 and coverage_enh >= 0.35:
        # Confidence based on coverage and segments
        coverage_conf = min(1.0, coverage_enh / 0.5)
        segment_conf = min(1.0, num_segments_enh / 7.0)
        conf = (coverage_conf + segment_conf) / 2
        confidences.append(('method_1b', max(0.4, conf)))
    # If coverage is very high, accept fewer segments (likely a solid or near-solid line)
    elif coverage_enh >= 0.90 and num_segments_enh >= 2:
        # Check the center rows/cols in ORIGINAL image for darkness
        # Use full width/height since dashed line dots may be at edges
        if not horizontal:
            center_strip_orig = gray[h // 4:3 * h // 4, w // 2 - 5:w // 2 + 5]
        else:
            center_strip_orig = gray[h // 2 - 5:h // 2 + 5, :]  # Full width for horizontal
        # Check for dark pixels indicating a dashed line
        if np.min(center_strip_orig) < 50:
            # Very high coverage, confirmed dark center
            conf = min(1.0, coverage_enh)
            confidences.append(('method_1b', max(0.5, conf)))

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
                    # Confidence based on regularity and number of spots
                    regularity = 1.0 - (gap_std / mean_gap)
                    num_spots_bonus = min(1.0, len(dark_positions) / 5.0)
                    conf = 0.5 + regularity * 0.3 + num_spots_bonus * 0.2
                    confidences.append(('method_2', conf))

    # Methods 3, 4, 7 PERMANENTLY REMOVED (confirmed dead code)
    # Testing showed disabling them had ZERO impact: 14/21 PASS unchanged
    # These blur-based methods never affected final decisions
    # Removed for code simplicity and maintainability

    # Define variables needed by remaining methods (5, 6)
    trim_pct = 0.25
    y_start = int(h * trim_pct)
    y_end = int(h * (1 - trim_pct))
    x_start = int(w * trim_pct)
    x_end = int(w * (1 - trim_pct))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Needed by Method 5

    # Method 5: Horizontal walls - check right portion
    if horizontal and w > 20 and not near_box_boundary:
        right_portion = blurred[h // 2 - 3:h // 2 + 3, w // 2:]
        right_min = int(np.min(right_portion))
        right_mean = np.mean(right_portion)
        if right_min < 30 and right_mean < 90:
            # Confidence based on darkness
            min_conf = (30 - right_min) / 30
            mean_conf = (90 - right_mean) / 90
            conf = (min_conf + mean_conf) / 2
            confidences.append(('method_5', max(0.4, conf)))

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

        if mean_freq_power > 0:
            ratio = max_freq_power / mean_freq_power
            std = np.std(profile)

            # Confidence based on ratio strength and std
            conf = 0.0
            if ratio > 3.0 and std > 15:
                # Scale confidence: ratio 3.0 = 0.5, ratio 6.0+ = 1.0
                conf = min(1.0, (ratio - 3.0) / 3.0 + 0.5)

            if conf > 0.2:
                confidences.append(('method_6', conf))

    # Method 7: Relaxed blur + enhancement - PERMANENTLY REMOVED (dead code)

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
            # Confidence based on darkness difference and std
            diff_conf = min(1.0, darkness_diff / 24)
            std_conf = min(1.0, profile_std / 24)
            conf = (diff_conf + std_conf) / 2
            confidences.append(('method_8', max(0.4, conf)))

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
                        confidences.append(('method_9', 1.0))

    # Ensemble voting using confidence scores
    # Require multiple methods to agree based on confidence levels
    if len(confidences) == 0:
        return False

    method_names = [m for m, c in confidences]
    conf_values = [c for m, c in confidences]

    num_methods = len(confidences)
    max_conf = max(conf_values)
    avg_conf = sum(conf_values) / num_methods

    # Decision logic based on number of agreeing methods:
    # Tuned thresholds to balance false positives and false negatives
    # threshold_mult allows dynamic adjustment: >1.0 = stricter, <1.0 = looser
    # Base thresholds: single=0.52, two=0.38, three+=0.28 (well-tuned, don't change)
    # Note: Tightening to 0.60/0.45/0.35 caused regression (Puzzle 21: 405→397)
    if num_methods == 1:
        # Single method must be confident
        is_boundary = max_conf > (0.52 * threshold_mult)
    elif num_methods == 2:
        # Two methods agreeing, use average confidence
        is_boundary = avg_conf > (0.38 * threshold_mult)
    elif num_methods >= 3:
        # Multiple methods agree - lenient threshold
        is_boundary = avg_conf > (0.28 * threshold_mult)
    else:
        is_boundary = False

    return is_boundary


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
    return add_ocr_border(gray, size=20, value=0 if binary else 255)


def ocr_cage_sum(corner_crop, reader_instance, cage_size=None, cell_h=200, cell_w=200):
    """OCR a cage sum from a corner crop using multiple strategies."""
    if corner_crop.size == 0:
        return None, 0

    all_results = []
    crops = [
        (0.35, 0.60, 4),  # Wider crop for two-digit numbers
        (0.35, 0.50, 4),
        (0.32, 0.40, 5),
        (0.32, 0.35, 5),
        (0.28, 0.30, 6),  # Slightly wider than 0.25 to avoid cutting off second digit
        (0.28, 0.25, 6),  # Keep original narrow crop
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

        # Try two approaches: OTSU binary and inverted grayscale
        images_to_try = []

        # Approach 1: OTSU binary (original)
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_to_try.append(add_ocr_border(binary, size=10, value=255))

        # Approach 2: Inverted grayscale (works better for small numbers)
        inverted = 255 - scaled
        images_to_try.append(add_ocr_border(inverted, size=10, value=0))

        for img in images_to_try:
            ocr_results = reader_instance.readtext(img, allowlist='0123456789', detail=1)

            for bbox, txt, conf in ocr_results:
                if txt.isdigit():
                    val = int(txt)
                    x_center = (bbox[0][0] + bbox[2][0]) / 2
                    x_ratio = x_center / img.shape[1]

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


def ocr_cage_sum_tesseract(corner_crop, cell_h=200, cell_w=200):
    """OCR cage sum using Tesseract PSM 7 (single line mode).

    Use as fallback when EasyOCR confidence is low.
    """
    if corner_crop.size == 0:
        return None, 0

    # Crop to top-left where cage sums typically appear
    h_frac, w_frac = 0.35, 0.50
    crop = corner_crop[5:int(cell_h * h_frac), 8:int(cell_w * w_frac)]
    if crop.size == 0:
        return None, 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Scale up for better OCR
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Apply OTSU threshold
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract PSM 7 = treat image as single text line
    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    try:
        text = pytesseract.image_to_string(binary, config=config).strip()
        if text.isdigit():
            val = int(text)
            if 1 <= val <= 45:
                return val, 0.85  # Give Tesseract results moderate confidence
    except Exception:
        pass

    return None, 0


def ocr_cage_sum_ensemble(corner_crop, reader_instance, cell_h=200, cell_w=200):
    """Ensemble OCR for cage sums with special handling for small digits.

    Combines EasyOCR and Tesseract with voting logic to handle:
    - Small digits (2, 8) often misread
    - Confusion between single and two-digit numbers (2 vs 28, 8 vs 18, etc.)
    """
    # Get EasyOCR result (primary)
    easy_val, easy_conf = ocr_cage_sum(corner_crop, reader_instance, cage_size=None, cell_h=cell_h, cell_w=cell_w)

    # Get Tesseract result (secondary/tiebreaker)
    tess_val, tess_conf = ocr_cage_sum_tesseract(corner_crop, cell_h=cell_h, cell_w=cell_w)

    # If both agree, return with high confidence
    if easy_val == tess_val and easy_val is not None:
        avg_conf = (easy_conf + tess_conf) / 2
        return easy_val, avg_conf

    # If EasyOCR has very high confidence and valid result, trust it
    if easy_val is not None and easy_conf > 0.90:
        return easy_val, easy_conf

    # Special handling for small digit confusion (2 vs 28, 8 vs 18, etc.)
    # If one engine reads single digit and other reads two-digit where first digit matches
    if easy_val is not None and tess_val is not None:
        # Check if one is single digit, other is two-digit with matching first digit
        easy_is_small = easy_val < 10
        tess_is_small = tess_val < 10

        if easy_is_small != tess_is_small:  # One single, one double-digit
            small_val = easy_val if easy_is_small else tess_val
            large_val = tess_val if easy_is_small else easy_val
            small_conf = easy_conf if easy_is_small else tess_conf
            large_conf = tess_conf if easy_is_small else easy_conf

            # Check if large number's first digit matches small number
            if large_val >= 10 and large_val // 10 == small_val:
                # Prefer two-digit if:
                # 1. Large number is in valid range and has decent confidence
                # 2. OR small number is commonly confused (2, 8)
                if large_val <= 45 and large_conf > 0.50:
                    return large_val, large_conf
                elif small_val in [2, 8]:
                    # These are commonly misread - prefer two-digit reading
                    if large_conf > 0.40:
                        return large_val, large_conf * 0.95

    # If EasyOCR has better confidence, use it
    if easy_val is not None and (tess_val is None or easy_conf > tess_conf + 0.15):
        return easy_val, easy_conf

    # If Tesseract has better confidence, use it
    if tess_val is not None and (easy_val is None or tess_conf > easy_conf):
        return tess_val, tess_conf

    # Default to EasyOCR if both are similar
    return easy_val, easy_conf


def try_tesseract_digit(cell_img, size=9):
    """Try Tesseract OCR for single digit recognition.

    Uses PSM 10 (single character) mode for digit detection.

    Args:
        cell_img: Grayscale cell image
        size: Grid size (for valid digit range)

    Returns:
        (digit, confidence) tuple, (0, 0) if no valid digit found
    """
    if cell_img is None or cell_img.size == 0:
        return 0, 0

    try:
        # Prepare image - resize and threshold
        h, w = cell_img.shape[:2]
        margin = int(min(h, w) * 0.12)
        center = cell_img[margin:h-margin, margin:w-margin]

        if center.size == 0:
            return 0, 0

        # Scale up for better OCR
        scaled = cv2.resize(center, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE + threshold
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scaled)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Auto-invert if needed (check corners)
        corner_size = max(5, binary.shape[0] // 10)
        corners = [
            binary[:corner_size, :corner_size],
            binary[:corner_size, -corner_size:],
            binary[-corner_size:, :corner_size],
            binary[-corner_size:, -corner_size:]
        ]
        if np.mean([np.mean(c) for c in corners]) > 128:
            binary = 255 - binary

        # Add border for Tesseract
        binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        # PSM 10: Treat image as single character
        config = '--psm 10 -c tessedit_char_whitelist=123456789'
        result = pytesseract.image_to_string(binary, config=config).strip()

        if result.isdigit() and 1 <= int(result) <= size:
            # Get confidence from detailed output
            try:
                data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
                confs = [int(c) for c in data['conf'] if str(c).lstrip('-').isdigit() and int(c) > 0]
                conf = max(confs) / 100.0 if confs else 0.6
            except Exception:
                conf = 0.6
            return int(result), conf

    except Exception:
        pass

    return 0, 0


def filter_grid_line_false_positives(right_walls, bottom_walls, size=9, threshold=6):
    """Remove boundaries that follow grid line patterns.

    Grid lines create distinctive patterns: they appear at ALL cells in a row/column
    at the same position index. Cage boundaries are sporadic.

    Args:
        right_walls: 2D boolean array of vertical boundaries
        bottom_walls: 2D boolean array of horizontal boundaries
        size: Grid size (6 or 9)
        threshold: Minimum number of cells with boundary to consider it a grid line
                   For 9x9: >6 means >67% of cells, for 6x6: >4 means >67%

    Returns:
        (right_walls, bottom_walls) with grid line false positives removed
    """
    # Check vertical boundaries (right_walls)
    # For each column position (0 to size-2), count how many rows have a boundary
    for col_idx in range(size - 1):
        boundary_count = sum(1 for r in range(size) if right_walls[r][col_idx])

        # If >threshold rows have a boundary at this column, it's likely a grid line
        if boundary_count > threshold:
            # Remove all boundaries at this column position
            for r in range(size):
                right_walls[r][col_idx] = False

    # Same for horizontal boundaries (bottom_walls)
    for row_idx in range(size - 1):
        boundary_count = sum(1 for c in range(size) if bottom_walls[row_idx][c])

        # If >threshold columns have a boundary at this row, it's likely a grid line
        if boundary_count > threshold:
            # Remove all boundaries at this row position
            for c in range(size):
                bottom_walls[row_idx][c] = False

    return right_walls, bottom_walls


def detect_cage_boundaries(warped_clean, x_bounds, y_bounds, size, threshold_mult=1.0):
    """Detect cage boundaries with adjustable sensitivity.

    Args:
        warped_clean: Preprocessed warped grid image
        x_bounds: X coordinates of cell boundaries
        y_bounds: Y coordinates of cell boundaries
        size: Grid size (6 or 9)
        threshold_mult: Multiplier for detection thresholds (>1.0 = stricter, <1.0 = looser)

    Returns:
        (right_walls, bottom_walls) tuple of 2D boolean arrays
    """
    # Box boundary positions
    if size == 9:
        box_cols = [2, 5]
        box_rows = [2, 5]
    else:
        box_cols = [1, 3]
        box_rows = [1, 3]

    right_walls = [[False] * size for _ in range(size)]
    bottom_walls = [[False] * size for _ in range(size)]

    for r in range(size):
        for c in range(size - 1):
            is_box_boundary = c in box_cols
            x = x_bounds[c + 1]
            y1, y2 = y_bounds[r], y_bounds[r + 1]
            crop = warped_clean[y1:y2, x - 25:x + 25]
            if is_cage_boundary(crop, horizontal=False, near_box_boundary=is_box_boundary, threshold_mult=threshold_mult):
                right_walls[r][c] = True
        right_walls[r][size - 1] = True

    for r in range(size - 1):
        for c in range(size):
            is_box_boundary = r in box_rows
            y = y_bounds[r + 1]
            x1, x2 = x_bounds[c], x_bounds[c + 1]
            crop = warped_clean[y - 25:y + 25, x1:x2]
            if is_cage_boundary(crop, horizontal=True, near_box_boundary=is_box_boundary, threshold_mult=threshold_mult):
                bottom_walls[r][c] = True
    bottom_walls[size - 1] = [True] * size

    return right_walls, bottom_walls


def solve_extraction(image_path, size=None, include_candidates=False, debug=False):
    """Main extraction function - extract killer sudoku from image.

    Args:
        image_path: Path to the image file
        size: Grid size (6 or 9), auto-detected if None
        include_candidates: If True, also extract pencil marks/candidates
        debug: If True, print debug info

    Returns:
        {
            "board": [[...], ...],
            "cage_map": [...],
            "cage_sums": {...},
            "candidates": [[[1,2,3], [], ...], ...] if include_candidates else None
        }
    """
    if size is None:
        size = 6 if "6x6" in image_path else 9

    reader = get_reader()
    warped = get_warped_grid(image_path)
    if warped is None:
        return None

    # Detect actual grid line positions (with fallback to uniform spacing)
    x_bounds, y_bounds = get_cell_boundaries(warped, size)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Remove thick grid lines for better cage boundary detection
    warped_clean = remove_grid_lines(warped, size)

    # ROBUST SOLUTION: Remove grid lines that span entire grid (>95% coverage)
    # This eliminates false positives from dailykillersudoku.com style grids
    if debug:
        print("Checking for full-grid lines to remove...")
    warped_clean = remove_full_grid_lines(warped_clean, size, debug=debug)

    # Detect cage boundaries with default threshold (1.0)
    # The threshold_mult parameter allows tuning: >1.0 = stricter, <1.0 = looser
    right_walls, bottom_walls = detect_cage_boundaries(warped_clean, x_bounds, y_bounds, size, threshold_mult=1.0)

    if debug:
        boundary_count = sum(sum(row) for row in right_walls) + sum(sum(row) for row in bottom_walls)
        print(f"Boundaries detected: {boundary_count}")

    # NOTE: Pattern-based filtering (>6/9 cells) was tested and FAILED
    # Result: 2/21 PASS (catastrophic - removed too many legitimate boundaries)
    # Root cause: Detection picks up BOTH grid residue AND cage boundaries at same positions
    # Cannot distinguish them post-detection. Filtering removes everything indiscriminately.

    # Detect anchor cells (cells with cage sum numbers)
    anchor_cells = set()
    anchor_sums = {}
    for r in range(size):
        for c in range(size):
            y1, y2 = y_bounds[r], y_bounds[r + 1]
            x1, x2 = x_bounds[c], x_bounds[c + 1]
            cell_h, cell_w = y2 - y1, x2 - x1
            cell_color = warped[y1:y2, x1:x2]
            detected_sum, conf = ocr_cage_sum_ensemble(cell_color, reader, cell_h=cell_h, cell_w=cell_w)
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

    # NOTE: Orphan cage merging logic was removed. The ML boundary classifier achieves
    # 90%+ test accuracy, so we trust ML boundary detections. If OCR misses a cage sum,
    # that's an OCR issue to fix, not a boundary issue. Orphan merging was incorrectly
    # removing valid boundaries when OCR failed to detect small sums.

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
    candidates = [[[] for _ in range(size)] for _ in range(size)] if include_candidates else None
    cage_sums = {}
    cage_map = [[None] * size for _ in range(size)]

    def get_id(i):
        return chr(ord('a') + i) if i < 26 else 'a' + chr(ord('a') + i - 26)

    sum_cell_to_idx = {c['sum_cell']: i for i, c in enumerate(cages)}

    for r in range(size):
        for c in range(size):
            y1, y2 = y_bounds[r], y_bounds[r + 1]
            x1, x2 = x_bounds[c], x_bounds[c + 1]
            cell_h, cell_w = y2 - y1, x2 - x1
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

                        v, _ = fix_digit_confusion(v, wb, hb)
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

            # If include_candidates and cell is empty, detect pencil marks
            if include_candidates and val == 0:
                cell_candidates = detect_candidates_in_cell(cell_img, reader, size)
                candidates[r][c] = cell_candidates
                if debug and cell_candidates:
                    print(f"Cell [{r},{c}]: candidates={cell_candidates}")

            # Cage sum
            if (r, c) in sum_cell_to_idx:
                idx = sum_cell_to_idx[(r, c)]
                cid = get_id(idx)
                predetected = cages[idx].get('predetected_sum')
                if predetected is not None:
                    cage_sums[cid] = predetected
                else:
                    detected_sum, conf = ocr_cage_sum_ensemble(cell_color, reader, cell_h=cell_h, cell_w=cell_w)
                    if detected_sum is not None:
                        cage_sums[cid] = detected_sum
                    else:
                        # Try Tesseract when EasyOCR fails
                        tess_sum, tess_conf = ocr_cage_sum_tesseract(cell_color, cell_h=cell_h, cell_w=cell_w)
                        if tess_sum is not None:
                            cage_sums[cid] = tess_sum
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
            # Two-digit misreads where digits merge: "11" -> "4", "14" -> "4", etc.
            (4, 11, 7),   # "4" misread from "11" (two 1s merged look like 4)
            (4, 17, 13),  # "4" misread from "17"
            (8, 11, 3),   # "8" misread from "11" (vertically stacked 1s)
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

    # Validate extraction before returning
    is_valid, issues = ExtractionValidator.validate_killer(board, cage_map, cage_sums, size)
    if not is_valid and debug:
        print(f"Validation issues: {issues}")

    # OPTION A: Post-process to fix common false positives
    # False positives are single-cell cages with very small sums (0, 1, 2)
    total_sum = sum(cage_sums.values())
    target_sum = 405 if size == 9 else 126

    if total_sum != target_sum and abs(total_sum - target_sum) <= 25:
        if debug:
            print(f"Sum {total_sum}/{target_sum} (diff={total_sum - target_sum}), attempting cleanup...")

        # Find single-cell cages with suspicious sums
        cage_to_cells = {}
        for r in range(size):
            for c in range(size):
                cage_id = cage_map[r][c]
                if cage_id not in cage_to_cells:
                    cage_to_cells[cage_id] = []
                cage_to_cells[cage_id].append((r, c))

        # Identify suspicious single-cell cages
        suspicious = []
        excess = total_sum - target_sum
        deficit = target_sum - total_sum

        for cage_id, cells in cage_to_cells.items():
            if len(cells) == 1 and cage_id in cage_sums:
                cage_sum = cage_sums[cage_id]

                # Suspicious if:
                # 1. Very small sum (0-3) - likely false positive
                # 2. If sum is over and this cage <= excess - could be part of split cage
                is_suspicious = (cage_sum <= 3) or (excess > 0 and cage_sum <= excess + 3)

                if is_suspicious:
                    suspicious.append((cage_id, cage_sum, cells[0]))

        if suspicious:
            if debug:
                print(f"Found {len(suspicious)} suspicious single-cell cages: {[(s[0], s[1]) for s in suspicious]}")

            # Merge suspicious cages with adjacent cages
            for cage_id, cage_sum, (r, c) in sorted(suspicious, key=lambda x: x[1]):
                # Find adjacent cages
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbor_id = cage_map[nr][nc]
                        if neighbor_id != cage_id and neighbor_id in cage_sums:
                            neighbors.append((neighbor_id, cage_sums[neighbor_id]))

                if neighbors:
                    # Merge with largest adjacent cage
                    merge_with = max(neighbors, key=lambda x: x[1])[0]

                    if debug:
                        print(f"  Merging cage {cage_id} (sum={cage_sum}) into {merge_with}")

                    # Update cage_map
                    cage_map[r][c] = merge_with

                    # Update cage_sums
                    if merge_with in cage_sums:
                        cage_sums[merge_with] += cage_sum
                    del cage_sums[cage_id]

                    # Recalculate total
                    total_sum = sum(cage_sums.values())

                    if total_sum == target_sum:
                        if debug:
                            print(f"  Target sum achieved!")
                        break

    # OPTION C: Check if sum == target, use Gemini fallback if not
    total_sum = sum(cage_sums.values())

    if total_sum != target_sum:
        if debug:
            print(f"Sum mismatch: {total_sum}/{target_sum}, attempting Gemini fallback...")

        # Try Gemini API as fallback
        gemini_result = extract_with_gemini_api(image_path, size, debug=debug)

        if gemini_result is not None:
            gemini_sum = sum(gemini_result['cage_sums'].values())
            if debug:
                print(f"Gemini sum: {gemini_sum}/{target_sum}")

            # If Gemini got it right, use Gemini result
            if gemini_sum == target_sum:
                if debug:
                    print("Using Gemini result (sum matches target)")
                result = gemini_result
                result["validation_issues"] = []
                result["fallback_used"] = "gemini"
                if include_candidates:
                    result["candidates"] = candidates
                return result
            else:
                # Neither worked, return local result with flags
                if debug:
                    print(f"Both local ({total_sum}) and Gemini ({gemini_sum}) failed to match target")

    # Return local result (either sum matches or fallback failed/unavailable)
    result = {
        "size": size,
        "board": board,
        "cage_map": cage_map,
        "cage_sums": cage_sums,
        "walls": {"right_walls": right_walls, "bottom_walls": bottom_walls},
        "validation_issues": issues if not is_valid else []
    }
    if include_candidates:
        result["candidates"] = candidates
    return result


def extract_with_gemini_api(image_path, size=9, debug=False):
    """Extract killer sudoku using Gemini API as fallback.

    Args:
        image_path: Path to the image
        size: Grid size (6 or 9)
        debug: Print debug info

    Returns:
        Extraction result dict or None if API call fails
    """
    try:
        import os
        import requests
        import base64
        import json

        # Get API key from environment
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            if debug:
                print("GOOGLE_API_KEY not set, skipping Gemini fallback")
            return None

        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Prepare prompt
        prompt = f"""Extract this {size}x{size} killer sudoku puzzle from the image.

Return ONLY a valid JSON object with this exact structure:
{{
  "board": [[0,0,0,...], ...],  // {size}x{size} grid, 0 for empty cells
  "cage_map": [["a","a","b",...], ...],  // {size}x{size} grid with cage IDs
  "cage_sums": {{"a": 12, "b": 15, ...}}  // Map of cage ID to sum
}}

Rules:
- Cage sums must total {405 if size == 9 else 126}
- Each cage must be contiguous
- Use lowercase letters for cage IDs
- Return ONLY the JSON, no other text"""

        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_data}}
                ]
            }]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            if debug:
                print(f"Gemini API error: {response.status_code}")
            return None

        resp_json = response.json()
        if 'candidates' not in resp_json or len(resp_json['candidates']) == 0:
            if debug:
                print("No candidates in Gemini response")
            return None

        text = resp_json['candidates'][0]['content']['parts'][0]['text']

        # Extract JSON from response (might have markdown code fences)
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        # Clean control characters and other problematic characters
        import re
        # Remove control characters except newline, carriage return, tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Try to extract JSON object if response has extra text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        # Parse JSON
        result = json.loads(text)

        # Add size field
        result['size'] = size

        if debug:
            print(f"Gemini extraction successful")

        return result

    except Exception as e:
        if debug:
            print(f"Gemini extraction failed: {e}")
        return None


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


def extract_with_improvements(image_path, size=None, include_candidates=False, debug=False, use_vertex_fallback=True):
    """Main extraction function with all improvements."""
    if size is None:
        size = 6 if "6x6" in image_path else 9

    # Step 1: Initial extraction
    result = solve_extraction(image_path, size=size, include_candidates=include_candidates, debug=debug)
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
# Classic Sudoku Extraction - Helper Functions
# =============================================================================

def analyze_cell_characteristics(cell_img):
    """Analyze cell image to determine content characteristics.

    Returns dict with:
        - mean_intensity: average pixel value
        - contrast_ratio: ratio of dark to light pixels
        - has_content: whether cell has significant content
        - is_dark_on_light: True if content is dark on light background
    """
    if cell_img.size == 0:
        return {'mean_intensity': 255, 'contrast_ratio': 0, 'has_content': False, 'is_dark_on_light': True}

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img

    # Compute histogram to determine background vs foreground
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Find background (most common intensity region)
    # Typically either very light (>200) or very dark (<50)
    light_bg = np.sum(hist[200:])
    dark_bg = np.sum(hist[:50])

    is_dark_on_light = light_bg > dark_bg

    mean_intensity = np.mean(gray)

    # Compute contrast ratio
    if is_dark_on_light:
        dark_pixels = np.sum(gray < 120)
        light_pixels = np.sum(gray > 180)
    else:
        dark_pixels = np.sum(gray < 80)
        light_pixels = np.sum(gray > 150)

    total = gray.size
    contrast_ratio = dark_pixels / total if is_dark_on_light else light_pixels / total

    # Check if there's significant content
    has_content = contrast_ratio > 0.01

    return {
        'mean_intensity': mean_intensity,
        'contrast_ratio': contrast_ratio,
        'has_content': has_content,
        'is_dark_on_light': is_dark_on_light
    }


def find_largest_connected_component(binary_img):
    """Find the largest connected component in a binary image.

    Returns:
        - mask: binary mask of largest component
        - stats: dict with area, centroid_x, centroid_y, bbox (x, y, w, h)
        - None, None if no components found
    """
    # Ensure binary image
    if binary_img is None or binary_img.size == 0:
        return None, None

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    if num_labels <= 1:  # Only background
        return None, None

    # Find largest non-background component
    # stats columns: [left, top, width, height, area]
    # Skip label 0 (background)
    largest_label = 1
    largest_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    if largest_area == 0:
        return None, None

    # Create mask for largest component
    mask = (labels == largest_label).astype(np.uint8) * 255

    component_stats = {
        'area': largest_area,
        'centroid_x': centroids[largest_label][0],
        'centroid_y': centroids[largest_label][1],
        'bbox': (
            stats[largest_label, cv2.CC_STAT_LEFT],
            stats[largest_label, cv2.CC_STAT_TOP],
            stats[largest_label, cv2.CC_STAT_WIDTH],
            stats[largest_label, cv2.CC_STAT_HEIGHT]
        ),
        'num_components': num_labels - 1  # Exclude background
    }

    return mask, component_stats


def compute_edge_sharpness(cell_img):
    """Compute edge sharpness using Laplacian variance.

    Higher values indicate cleaner/bolder strokes (placed digits).
    Lower values indicate fuzzier/lighter strokes (pencil marks).
    """
    if cell_img is None or cell_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Variance of Laplacian indicates edge sharpness
    return laplacian.var()


def is_placed_digit(cell_gray, debug=False):
    """Determine if cell contains a placed digit (not pencil marks).

    Uses multiple criteria:
    1. Size: placed digits have meaningful pixel coverage
    2. Position: placed digits are centered
    3. Contiguity: placed digits form connected components (not scattered)
    4. Bounding box: placed digits have proper height

    Returns: (is_placed, confidence, reason)
    """
    if cell_gray is None or cell_gray.size == 0:
        return False, 0.0, "empty"

    h, w = cell_gray.shape

    # Detect if dark background based on cell mean
    is_dark_background = np.mean(cell_gray) < 100

    # Trim margins to avoid grid lines
    # Use larger margin for dark backgrounds where grid lines are more prominent
    margin_pct = 0.12 if is_dark_background else 0.08
    margin = int(min(h, w) * margin_pct)
    trimmed = cell_gray[margin:h-margin, margin:w-margin]

    if trimmed.size == 0:
        return False, 0.0, "trim_failed"

    # Use OTSU threshold - detect if light or dark background
    mean_val = np.mean(trimmed)

    # Apply OTSU threshold
    _, binary = cv2.threshold(trimmed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If light background (mean > 128), digits are dark, so invert
    # If dark background (mean < 128), digits are light, binary is already correct
    if mean_val > 128:
        binary = 255 - binary

    # Find largest connected component
    mask, stats = find_largest_connected_component(binary)

    if mask is None or stats is None:
        return False, 0.0, "no_component"

    trimmed_area = trimmed.shape[0] * trimmed.shape[1]

    # Criterion 1: Size ratio - digits vary widely in stroke thickness
    size_ratio = stats['area'] / trimmed_area

    # Very thin-stroked digits might only be 1.5% of cell area
    if size_ratio < 0.015:
        return False, 0.0, f"too_small:{size_ratio:.3f}"
    if size_ratio > 0.70:
        return False, 0.0, f"too_large:{size_ratio:.3f}"

    # Criterion 2: Centering
    center_x = trimmed.shape[1] / 2
    center_y = trimmed.shape[0] / 2

    dist_x = abs(stats['centroid_x'] - center_x) / center_x
    dist_y = abs(stats['centroid_y'] - center_y) / center_y

    # Placed digits should be reasonably centered
    # Be slightly more lenient on vertical centering (some digits are top/bottom heavy)
    if dist_x > 0.60 or dist_y > 0.65:
        return False, 0.0, f"off_center:x={dist_x:.2f},y={dist_y:.2f}"

    # Criterion 3: Bounding box height is key - placed digits are tall
    bbox = stats['bbox']  # (x, y, w, h)
    bbox_h_ratio = bbox[3] / trimmed.shape[0]
    bbox_w_ratio = bbox[2] / trimmed.shape[1]

    # Placed digits should cover significant vertical portion of cell
    # This is the key differentiator from pencil marks
    if bbox_h_ratio < 0.35:
        return False, 0.0, f"bbox_short:{bbox_h_ratio:.2f}"

    # Also check that bbox is somewhat centered
    bbox_center_x = (bbox[0] + bbox[2]/2) / trimmed.shape[1]
    bbox_center_y = (bbox[1] + bbox[3]/2) / trimmed.shape[0]

    if abs(bbox_center_x - 0.5) > 0.35 or abs(bbox_center_y - 0.5) > 0.35:
        return False, 0.0, f"bbox_off_center:x={bbox_center_x:.2f},y={bbox_center_y:.2f}"

    # Criterion 4: Component count - stricter for pencil mark detection
    num_components = stats['num_components']

    # Reject if too many scattered components (likely pencil marks)
    # Allow up to 7 components for noisy/fragmented digits
    if num_components > 7:
        return False, 0.0, f"scattered:{num_components}_components"

    # Special check: if bbox spans almost full height AND multiple components,
    # it's likely pencil marks scattered across the cell rather than one digit
    if bbox_h_ratio > 0.85 and num_components > 2:
        return False, 0.0, f"scattered_tall:{num_components}_components,h={bbox_h_ratio:.2f}"

    # Compute confidence based on how well criteria are met
    size_score = min(1.0, size_ratio / 0.10)  # Higher size = better
    center_score = 1.0 - (dist_x + dist_y) / 2
    height_score = min(1.0, bbox_h_ratio / 0.6)  # Optimal around 60% height
    component_score = 1.0 if num_components <= 3 else 0.7 if num_components <= 5 else 0.4

    confidence = (size_score + center_score + height_score + component_score) / 4
    confidence = max(0.0, min(1.0, confidence))

    return True, confidence, f"ok:size={size_ratio:.2f},h={bbox_h_ratio:.2f},components={num_components}"


def prepare_cell_hsv(cell_color, scale=2):
    """Use HSV Value channel for background-independent thresholding.

    This handles highlighted/colored cells better than grayscale by
    using the brightness (Value) channel which ignores hue.
    """
    if cell_color.size == 0:
        return None

    resized = cv2.resize(cell_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if len(resized.shape) == 2:
        # Already grayscale
        v = resized
        highlight_detected = False
    else:
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Detect highlighting (high saturation + varying value)
        # Blue highlighting: H ~100-130, S > 50, V > 100
        highlight_mask = cv2.inRange(hsv, (80, 40, 80), (140, 255, 255))
        highlight_ratio = np.sum(highlight_mask > 0) / highlight_mask.size
        highlight_detected = highlight_ratio > 0.05

        # If highlighted, apply CLAHE to enhance contrast
        if highlight_detected:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            v = clahe.apply(v)

    # Apply OTSU on Value channel
    _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine if we need to invert based on corner samples
    # (corners are typically background)
    h, w = binary.shape
    corner_size = max(5, h // 10)
    corners = [
        binary[:corner_size, :corner_size],
        binary[:corner_size, -corner_size:],
        binary[-corner_size:, :corner_size],
        binary[-corner_size:, -corner_size:]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If corners are bright, background is white, digits are black - invert
    if corner_mean > 128:
        binary = 255 - binary

    return add_ocr_border(binary)


def prepare_for_classic_ocr(crop, scale=2, cell_color=None):
    """Prepare image crop for classic sudoku OCR.

    Uses OTSU thresholding with automatic background detection.
    Handles both light and dark background images.
    If cell_color is provided and standard method fails, falls back to HSV.
    """
    if crop.size == 0:
        return crop

    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Detect background type using corner sampling for more robust detection
    h, w = gray.shape
    corner_size = max(5, h // 10)
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])

    # Apply OTSU threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure digit is white on black background for OCR
    # Use corner-based detection for more robust inversion decision
    if corner_mean > 128:
        binary = 255 - binary

    # Add border for OCR
    return add_ocr_border(binary)


# =============================================================================
# Classic Sudoku Extraction - Main Function
# =============================================================================

def detect_candidates_in_cell(cell_img, reader, size=9):
    """Detect pencil mark candidates in a cell.

    Uses hierarchical content classification from PDF Section 5:
    - Solved Digit: Central, large (60-80% cell height)
    - Candidates: Corner positions, small (< 50% cell height)

    Returns: list of candidate digits (1-9)
    """
    h, w = cell_img.shape

    # Threshold the cell
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cell_img)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    candidate_regions = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        # Skip noise (too small)
        if bh < h * 0.08 or bw < w * 0.05:
            continue

        # Skip solved digit (too large - centered large digit)
        if bh > h * 0.50:
            continue

        # Skip if in the center (might be part of solved digit)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        if 0.35 < cx < 0.65 and 0.35 < cy < 0.65 and bh > h * 0.30:
            continue

        # Store region for OCR
        candidate_regions.append((x, y, bw, bh, cx, cy))

    # OCR each candidate region
    for x, y, bw, bh, cx, cy in candidate_regions:
        # Add margin
        margin = 3
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)

        roi = cell_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Scale up for better OCR
        scale = max(1, 28 // min(roi.shape[0], roi.shape[1]))
        if scale > 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Prepare for OCR
        roi_enhanced = cv2.equalizeHist(roi)
        _, roi_binary = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        try:
            results = reader.readtext(roi_binary, allowlist='123456789', detail=1)
            for _, txt, conf in results:
                if txt.isdigit() and conf > 0.3:
                    digit = int(txt)
                    if 1 <= digit <= size and digit not in candidates:
                        candidates.append(digit)
        except:
            pass

    return sorted(candidates)


def extract_classic_sudoku(image_path, size=9, include_candidates=False, debug=False):
    """Extract classic sudoku board from image (no cages).

    This handles puzzles with pencil marks by:
    1. Pre-filtering with is_placed_digit() to check for centered, tall content
    2. Running OCR with size and position filters
    3. Validating results with quality scoring

    Now includes image quality classification to route preprocessing
    appropriately for dark mode, low contrast, and other challenging images.

    Args:
        image_path: Path to the image file
        size: Grid size (6 or 9)
        include_candidates: If True, also extract pencil marks/candidates
        debug: If True, print debug info

    Returns:
        {
            "board": [[...], ...],
            "candidates": [[[1,2,3], [], ...], ...] if include_candidates else None,
            "diagnostics": {...} if debug else None
        }
    """
    reader = get_reader()

    # Load image for quality classification before warping
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        return None

    # Classify image quality and get preprocessing parameters
    image_quality, quality_metrics = classify_image_quality(raw_img)
    preprocess_params = get_preprocessing_params(image_quality)

    if debug:
        print(f"Image quality: {image_quality}")
        print(f"Metrics: {quality_metrics}")
        print(f"Params: {preprocess_params}")

    warped = get_warped_grid(image_path)
    if warped is None:
        return None

    cell_h, cell_w = 1800 // size, 1800 // size

    # Standard grayscale conversion - dark mode preprocessing is applied per-cell as fallback
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Apply sharpening if needed for blurry images
    if preprocess_params['apply_sharpening']:
        gray = apply_sharpening(gray)

    board = [[0] * size for _ in range(size)]
    candidates = [[[] for _ in range(size)] for _ in range(size)] if include_candidates else None

    # Get CNN classifier (may be None if not available)
    cnn_classifier = get_cnn_classifier()

    def try_cnn(cell_img):
        """Try CNN digit classification first.

        Returns (digit, confidence) or (0, 0) if CNN unavailable or low confidence.
        """
        if cnn_classifier is None:
            return 0, 0

        h, w = cell_img.shape

        # Trim borders (same as EasyOCR path)
        margin = int(min(h, w) * 0.12)
        center = cell_img[margin:h-margin, margin:w-margin]

        if center.size == 0:
            return 0, 0

        # CNN expects the cell as-is, it handles preprocessing internally
        digit, conf = cnn_classifier.predict(center)

        # Valid digit range check
        if digit < 0 or digit > size:
            return 0, 0

        return digit, conf

    def try_easyocr(cell_img, min_height_ratio=0.15):
        """Run EasyOCR on a cell and return best digit with quality score."""
        h, w = cell_img.shape

        # Trim borders
        margin = int(min(h, w) * 0.12)
        center = cell_img[margin:h-margin, margin:w-margin]

        if center.size == 0:
            return 0, 0

        # Prepare for OCR
        ocr_input = prepare_for_classic_ocr(center)
        ocr_h, ocr_w = ocr_input.shape[:2]

        # Run EasyOCR
        results = reader.readtext(ocr_input, allowlist='0123456789', detail=1)

        best_digit = 0
        best_score = 0

        for bbox, txt, conf in results:
            if not txt.isdigit():
                continue

            digit = int(txt)
            if digit < 1 or digit > size:
                continue

            # Get bounding box info
            bbox_h = abs(bbox[2][1] - bbox[0][1])
            bbox_w = abs(bbox[1][0] - bbox[0][0])
            bbox_cx = (bbox[0][0] + bbox[2][0]) / 2
            bbox_cy = (bbox[0][1] + bbox[2][1]) / 2

            height_ratio = bbox_h / ocr_h
            center_x = bbox_cx / ocr_w
            center_y = bbox_cy / ocr_h

            # Height filter - key for rejecting pencil marks
            if height_ratio < min_height_ratio:
                continue

            # Center filter (relaxed to allow edge-positioned small digits)
            if center_x < 0.15 or center_x > 0.85 or center_y < 0.15 or center_y > 0.85:
                continue

            # Quality score (prioritize confidence for small digits)
            height_score = max(0, (height_ratio - min_height_ratio) / (1 - min_height_ratio))
            center_dist = ((center_x - 0.5)**2 + (center_y - 0.5)**2) ** 0.5
            center_score = max(0, 1.0 - center_dist * 2)

            score = conf * 0.6 + height_score * 0.15 + center_score * 0.25

            # 1<->7 aspect ratio discrimination
            digit, _ = fix_digit_confusion(digit, bbox_w, bbox_h)

            if score > best_score:
                best_score = score
                best_digit = digit

        return best_digit, best_score

    def try_ocr(cell_img, min_height_ratio=0.15):
        """2-engine primary with Tesseract tiebreaker: CNN + EasyOCR, Tesseract on disagreement.

        Strategy:
        - If CNN highly confident (>0.95) AND detects digit, use CNN
        - If CNN and EasyOCR agree, use that result
        - If they disagree, use Tesseract as tiebreaker
        - If all 3 disagree, be conservative (prefer empty/lower digit)
        """
        # Get CNN prediction (fast, trained on task)
        cnn_digit, cnn_conf = try_cnn(cell_img)

        # If CNN is very confident, trust it directly
        if cnn_digit > 0 and cnn_conf > 0.95:
            return cnn_digit, cnn_conf

        # Get EasyOCR prediction
        easyocr_digit, easyocr_conf = try_easyocr(cell_img, min_height_ratio)

        # If both agree, use that result
        if cnn_digit == easyocr_digit:
            avg_conf = (cnn_conf + easyocr_conf) / 2
            return cnn_digit, avg_conf

        # Disagreement - use Tesseract as tiebreaker (but it's slower)
        tess_digit, tess_conf = try_tesseract_digit(cell_img, size)

        # Count votes
        votes = {}
        for digit in [cnn_digit, easyocr_digit, tess_digit]:
            votes[digit] = votes.get(digit, 0) + 1

        # Find majority
        max_votes = max(votes.values())
        winners = [d for d, v in votes.items() if v == max_votes]

        if max_votes >= 2:
            # At least 2/3 agree
            winner = winners[0]
            # Get confidences of agreeing engines
            confs = []
            if cnn_digit == winner:
                confs.append(cnn_conf)
            if easyocr_digit == winner:
                confs.append(easyocr_conf)
            if tess_digit == winner:
                confs.append(tess_conf)
            avg_conf = sum(confs) / len(confs) if confs else 0.5
            return winner, avg_conf * 0.9

        else:
            # All 3 disagree - be very conservative
            # Prefer empty (0) if any engine says empty
            if 0 in winners:
                return 0, 0

            # Otherwise, use highest confidence but penalize heavily
            best = max(
                [(cnn_digit, cnn_conf), (easyocr_digit, easyocr_conf), (tess_digit, tess_conf)],
                key=lambda x: x[1]
            )
            # Only accept if very confident despite disagreement
            if best[1] > 0.9:
                return best[0], best[1] * 0.5
            return 0, 0

    for r in range(size):
        for c in range(size):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell_img = gray[y1:y2, x1:x2]

            # Pre-filter with is_placed_digit
            is_placed, placement_conf, reason = is_placed_digit(cell_img)

            # Handle "too_large" as possible highlighted/selected cell
            # These cells often have colored backgrounds that confuse thresholding
            is_possibly_highlighted = reason.startswith("too_large")

            if not is_placed and not is_possibly_highlighted:
                if debug:
                    print(f"Cell [{r},{c}]: skipped ({reason})")
                board[r][c] = 0
                # If include_candidates, try to detect pencil marks in empty cells
                if include_candidates:
                    cell_candidates = detect_candidates_in_cell(cell_img, reader, size)
                    candidates[r][c] = cell_candidates
                    if debug and cell_candidates:
                        print(f"Cell [{r},{c}]: candidates={cell_candidates}")
                continue

            # Run OCR
            digit, score = try_ocr(cell_img)

            # If low score, try with dilation
            if digit == 0 or score < 0.4:
                h, w = cell_img.shape
                margin = int(min(h, w) * 0.12)
                center = cell_img[margin:h-margin, margin:w-margin]
                ocr_input = prepare_for_classic_ocr(center)
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(ocr_input, kernel, iterations=3)

                results = reader.readtext(dilated, allowlist='0123456789', detail=1)
                for bbox, txt, conf in results:
                    if txt.isdigit():
                        d = int(txt)
                        bbox_h = abs(bbox[2][1] - bbox[0][1])
                        bbox_w = abs(bbox[1][0] - bbox[0][0])
                        height_ratio = bbox_h / dilated.shape[0]
                        if height_ratio >= 0.30 and 1 <= d <= size and conf > 0.5:
                            # 1<->7 aspect ratio discrimination
                            d, _ = fix_digit_confusion(d, bbox_w, bbox_h)
                            if conf > score:
                                digit, score = d, conf
                            break

            # If still no result, try HSV preprocessing (handles dark themes, colored backgrounds)
            if digit == 0 or score < 0.4:
                cell_color = warped[y1:y2, x1:x2]
                hsv_input = prepare_cell_hsv(cell_color)
                if hsv_input is not None:
                    results = reader.readtext(hsv_input, allowlist='0123456789', detail=1)
                    for bbox, txt, conf in results:
                        if txt.isdigit():
                            d = int(txt)
                            bbox_h = abs(bbox[2][1] - bbox[0][1])
                            bbox_w = abs(bbox[1][0] - bbox[0][0])
                            height_ratio = bbox_h / hsv_input.shape[0]
                            if height_ratio >= 0.30 and 1 <= d <= size and conf > 0.5:
                                # 1<->7 aspect ratio discrimination
                                d, _ = fix_digit_confusion(d, bbox_w, bbox_h)
                                if conf > score:
                                    digit, score = d, conf
                                break

            # Use dynamic confidence threshold from preprocessing params
            conf_threshold = preprocess_params['confidence_threshold']
            if score < conf_threshold:
                digit = 0

            if debug:
                print(f"Cell [{r},{c}]: digit={digit}, score={score:.2f}, threshold={conf_threshold:.2f}")

            board[r][c] = digit

            # If include_candidates and cell is empty, try to detect pencil marks
            if include_candidates and digit == 0:
                cell_candidates = detect_candidates_in_cell(cell_img, reader, size)
                candidates[r][c] = cell_candidates
                if debug and cell_candidates:
                    print(f"Cell [{r},{c}]: candidates={cell_candidates}")

    # Validate extraction before returning
    is_valid, issues = ExtractionValidator.validate_classic(board, size)
    if not is_valid and debug:
        print(f"Validation issues: {issues}")

    result = {"board": board, "validation_issues": issues if not is_valid else []}
    if include_candidates:
        result["candidates"] = candidates

    # Add diagnostics if debug mode
    if debug:
        result["diagnostics"] = {
            "image_quality": image_quality,
            "quality_metrics": quality_metrics,
            "preprocessing_params": preprocess_params
        }

    return result


# =============================================================================
# Extraction Validation
# =============================================================================

class ExtractionValidator:
    """Validate extracted puzzle data."""

    @staticmethod
    def validate_killer(board, cage_map, cage_sums, size=9):
        """Validate killer sudoku extraction.

        Returns: (is_valid, issues_list)
        """
        issues = []
        expected_sum = 405 if size == 9 else 126

        # Check total cage sum
        total = sum(cage_sums.values())
        if total != expected_sum:
            diff = expected_sum - total
            issues.append(f"sum_mismatch: {total}/{expected_sum} (diff={diff})")

        # Check for missing cage sums
        cage_ids = set()
        for row in cage_map:
            cage_ids.update(row)

        for cid in cage_ids:
            if cid not in cage_sums or cage_sums[cid] == 0:
                issues.append(f"missing_sum: cage {cid}")

        # Check cage sum ranges (1-45 for 9x9)
        max_sum = 45 if size == 9 else 21
        for cid, s in cage_sums.items():
            if s < 1 or s > max_sum:
                issues.append(f"invalid_sum: cage {cid} = {s}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_classic(board, size=9):
        """Validate classic sudoku extraction.

        Returns: (is_valid, issues_list)
        """
        issues = []

        # Check for invalid digits
        for r in range(size):
            for c in range(size):
                if board[r][c] < 0 or board[r][c] > size:
                    issues.append(f"invalid_digit: [{r},{c}] = {board[r][c]}")

        # Check for obvious duplicates in rows/cols/boxes
        for r in range(size):
            row_vals = [board[r][c] for c in range(size) if board[r][c] > 0]
            if len(row_vals) != len(set(row_vals)):
                issues.append(f"duplicate_in_row: {r}")

        for c in range(size):
            col_vals = [board[r][c] for r in range(size) if board[r][c] > 0]
            if len(col_vals) != len(set(col_vals)):
                issues.append(f"duplicate_in_col: {c}")

        # Check boxes
        box_size = 3 if size == 9 else 2
        for box_r in range(0, size, box_size):
            for box_c in range(0, size, box_size):
                box_vals = []
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        if board[r][c] > 0:
                            box_vals.append(board[r][c])
                if len(box_vals) != len(set(box_vals)):
                    issues.append(f"duplicate_in_box: ({box_r},{box_c})")

        return len(issues) == 0, issues

    @staticmethod
    def find_constraint_violations(board, size=9):
        """Find cells that violate sudoku constraints.

        Returns list of (row, col, digit, violation_type) for cells that
        have duplicate values in their row, column, or box.
        """
        violations = []
        box_size = 3 if size == 9 else 2

        # Check rows for duplicates
        for r in range(size):
            seen = {}
            for c in range(size):
                d = board[r][c]
                if d > 0:
                    if d in seen:
                        # Both cells have the same digit - both are violations
                        violations.append((r, c, d, 'row'))
                        prev_c = seen[d]
                        violations.append((r, prev_c, d, 'row'))
                    seen[d] = c

        # Check columns for duplicates
        for c in range(size):
            seen = {}
            for r in range(size):
                d = board[r][c]
                if d > 0:
                    if d in seen:
                        violations.append((r, c, d, 'col'))
                        prev_r = seen[d]
                        violations.append((prev_r, c, d, 'col'))
                    seen[d] = r

        # Check boxes for duplicates
        for box_r in range(0, size, box_size):
            for box_c in range(0, size, box_size):
                seen = {}
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        d = board[r][c]
                        if d > 0:
                            if d in seen:
                                violations.append((r, c, d, 'box'))
                                prev_r, prev_c = seen[d]
                                violations.append((prev_r, prev_c, d, 'box'))
                            seen[d] = (r, c)

        # Deduplicate violations (same cell might violate multiple constraints)
        unique = {}
        for r, c, d, vtype in violations:
            key = (r, c)
            if key not in unique:
                unique[key] = (r, c, d, [vtype])
            else:
                unique[key][3].append(vtype)

        return [(r, c, d, vtypes) for (r, c, d, vtypes) in unique.values()]


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
        - image: file
        - size: 6 or 9
        - include_candidates: "true" or "false" (default: false)
    Response: JSON with board, cage_map, cage_sums, and optionally candidates
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    size = request.form.get('size', '9')
    include_candidates = request.form.get('include_candidates', 'false').lower() == 'true'

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
            include_candidates=include_candidates,
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

        # Include candidates if requested
        if include_candidates:
            response["candidates"] = result.get("candidates", [[[] for _ in range(size)] for _ in range(size)])

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
        - image: file
        - size: 6 or 9
        - include_candidates: "true" or "false" (default: false)
    Response: JSON with board and optionally candidates
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    size = request.form.get('size', '9')
    include_candidates = request.form.get('include_candidates', 'false').lower() == 'true'

    try:
        size = int(size)
    except ValueError:
        size = 9

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = extract_classic_sudoku(
            tmp_path,
            size=size,
            include_candidates=include_candidates,
            debug=False
        )

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
