"""
Extract boundary training data from ground truth killer sudoku puzzles.

For each puzzle:
1. Load image and ground truth cage_map
2. Extract crops at all potential boundary positions (144 per 9×9 puzzle)
3. Label each crop as boundary (1) or non-boundary (0) using cage_map
4. Extract feature vectors from crops
5. Save to training dataset

Output:
- training_data/boundary_crops/positive/*.png (boundary examples)
- training_data/boundary_crops/negative/*.png (non-boundary examples)
- training_data/boundary_features.npz (feature vectors + labels)
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/cody/git/src/github.com/codyseavey/sudoku/extraction_service')
from app import get_warped_grid, get_cell_boundaries, remove_grid_lines, remove_full_grid_lines

def extract_boundary_labels_from_ground_truth(cage_map, size=9):
    """
    Determine which boundaries are cage boundaries from ground truth.

    Returns:
        right_walls: 2D boolean array [size x (size-1)]
        bottom_walls: 2D boolean array [(size-1) x size]
    """
    right_walls = [[False] * (size - 1) for _ in range(size)]
    bottom_walls = [[False] * size for _ in range(size - 1)]

    # Vertical boundaries (right walls)
    for r in range(size):
        for c in range(size - 1):
            if cage_map[r][c] != cage_map[r][c + 1]:
                right_walls[r][c] = True

    # Horizontal boundaries (bottom walls)
    for r in range(size - 1):
        for c in range(size):
            if cage_map[r][c] != cage_map[r + 1][c]:
                bottom_walls[r][c] = True

    return right_walls, bottom_walls

def extract_features_from_crop(crop, horizontal=False):
    """
    Extract feature vector from boundary crop.

    Features (38 total):
    - Basic stats (4): mean, std, min, max
    - Edge detection (3): Canny edge count, edge density, edge coverage
    - Gradient (4): Sobel X/Y magnitude mean, std
    - FFT (6): peak frequency power, mean frequency power, ratio, dominant frequency
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
    center_strip = gray[h//4:3*h//4, w//4:3*w//4] if not horizontal else gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_strip)
    center_min = np.min(center_strip)
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

def process_puzzle(image_path, json_path, size=9, output_dir='training_data'):
    """Extract all boundary crops and features from one puzzle."""
    # Load ground truth
    with open(json_path, 'r') as f:
        data = json.load(f)
    cage_map = data['cage_map']

    # Get ground truth boundaries
    gt_right_walls, gt_bottom_walls = extract_boundary_labels_from_ground_truth(cage_map, size)

    # Process image
    warped = get_warped_grid(image_path)
    if warped is None:
        print(f"Failed to warp: {image_path}")
        return None

    x_bounds, y_bounds = get_cell_boundaries(warped, size)
    warped_clean = remove_grid_lines(warped, size)
    warped_clean = remove_full_grid_lines(warped_clean, size)

    # Extract crops and features
    crops_positive = []
    crops_negative = []
    features_positive = []
    features_negative = []

    # Vertical boundaries
    for r in range(size):
        for c in range(size - 1):
            x = x_bounds[c + 1]
            y1, y2 = y_bounds[r], y_bounds[r + 1]
            crop = warped_clean[y1:y2, x - 25:x + 25]

            if crop.size > 0:
                features = extract_features_from_crop(crop, horizontal=False)
                is_boundary = gt_right_walls[r][c]

                if is_boundary:
                    crops_positive.append(crop)
                    features_positive.append(features)
                else:
                    crops_negative.append(crop)
                    features_negative.append(features)

    # Horizontal boundaries
    for r in range(size - 1):
        for c in range(size):
            y = y_bounds[r + 1]
            x1, x2 = x_bounds[c], x_bounds[c + 1]
            crop = warped_clean[y - 25:y + 25, x1:x2]

            if crop.size > 0:
                features = extract_features_from_crop(crop, horizontal=True)
                is_boundary = gt_bottom_walls[r][c]

                if is_boundary:
                    crops_positive.append(crop)
                    features_positive.append(features)
                else:
                    crops_negative.append(crop)
                    features_negative.append(features)

    return {
        'crops_positive': crops_positive,
        'crops_negative': crops_negative,
        'features_positive': features_positive,
        'features_negative': features_negative
    }

def main():
    """Extract training data from all test puzzles."""
    test_dir = Path('/home/cody/git/src/github.com/codyseavey/sudoku/test_data/killer_sudoku/9x9')
    output_dir = Path('/home/cody/git/src/github.com/codyseavey/sudoku/extraction_service/training_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'boundary_crops' / 'positive').mkdir(parents=True, exist_ok=True)
    (output_dir / 'boundary_crops' / 'negative').mkdir(parents=True, exist_ok=True)

    all_features_positive = []
    all_features_negative = []

    puzzle_files = sorted(test_dir.glob('*.png'))
    print(f"Processing {len(puzzle_files)} puzzles...")

    for puzzle_img in puzzle_files:
        puzzle_json = puzzle_img.with_suffix('.json')
        if not puzzle_json.exists():
            continue

        print(f"Processing {puzzle_img.name}...")
        result = process_puzzle(str(puzzle_img), str(puzzle_json), size=9, output_dir=str(output_dir))

        if result:
            # Save crop images
            puzzle_name = puzzle_img.stem
            for i, crop in enumerate(result['crops_positive']):
                cv2.imwrite(str(output_dir / 'boundary_crops' / 'positive' / f'{puzzle_name}_pos_{i}.png'), crop)
            for i, crop in enumerate(result['crops_negative']):
                cv2.imwrite(str(output_dir / 'boundary_crops' / 'negative' / f'{puzzle_name}_neg_{i}.png'), crop)

            # Collect features
            all_features_positive.extend(result['features_positive'])
            all_features_negative.extend(result['features_negative'])

            print(f"  Positive: {len(result['features_positive'])}, Negative: {len(result['features_negative'])}")

    # Combine and save
    X_positive = np.array(all_features_positive)
    X_negative = np.array(all_features_negative)
    y_positive = np.ones(len(X_positive))
    y_negative = np.zeros(len(X_negative))

    X = np.vstack([X_positive, X_negative])
    y = np.concatenate([y_positive, y_negative])

    # Save to file
    np.savez(output_dir / 'boundary_features.npz',
             X=X, y=y,
             feature_names=[
                 'mean', 'std', 'min', 'max',
                 'edge_count', 'edge_density', 'edge_coverage',
                 'gradient_mean', 'gradient_std', 'sobel_x_mean', 'sobel_y_mean',
                 'fft_max', 'fft_mean', 'fft_ratio', 'fft_dominant_freq', 'profile_std', 'profile_range',
                 'dark_ratio_80', 'very_dark_ratio_40', 'center_mean', 'center_min', 'background',
                 'contrast', 'moderate_dark_120', 'dark_60',
                 'line_profile_std', 'line_profile_min', 'line_profile_max', 'line_coverage',
                 'num_segments', 'num_transitions',
                 'opening_response', 'closing_response', 'total_area', 'total_perimeter',
                 'lbp_mean', 'lbp_std', 'lbp_max'
             ])

    print(f"\nTraining data saved:")
    print(f"  Total samples: {len(X)}")
    print(f"  Positive (boundaries): {len(X_positive)} ({len(X_positive)/len(X)*100:.1f}%)")
    print(f"  Negative (non-boundaries): {len(X_negative)} ({len(X_negative)/len(X)*100:.1f}%)")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Output: {output_dir / 'boundary_features.npz'}")

if __name__ == '__main__':
    main()
