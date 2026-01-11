#!/usr/bin/env python3
"""
Extract individual cells from sudoku test images for CNN training.

Extracts 200x200 (9x9) or 300x300 (6x6) cells from warped puzzle images,
using ground truth JSON files for labeling.
"""
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from app import get_warped_grid


# Files to exclude from training due to known issues (see test_data/KNOWN_ISSUES.md)
# Note: Most problematic files have been removed from test_data entirely
EXCLUDED_FILES = set()


def get_image_files(directory, exclude_files=None):
    """Get all image files in directory with matching JSON ground truth.

    Args:
        directory: Directory to search
        exclude_files: Set of file stems to exclude (e.g., {'9', '14'})
    """
    image_extensions = {'.png', '.jpg', '.jpeg'}
    images = []
    exclude_files = exclude_files or set()

    for f in os.listdir(directory):
        name, ext = os.path.splitext(f)
        if ext.lower() in image_extensions:
            # Check if this file should be excluded
            if name in exclude_files:
                print(f"  Excluding {f} (known issues)")
                continue

            json_path = os.path.join(directory, f"{name}.json")
            if os.path.exists(json_path):
                images.append((os.path.join(directory, f), json_path))

    return sorted(images)


def extract_cells(warped, size=9):
    """Extract individual cells from warped 1800x1800 grid.

    Args:
        warped: 1800x1800 BGR image
        size: Grid size (9 or 6)

    Returns:
        List of (cell_image, row, col) tuples
    """
    cell_size = 1800 // size
    cells = []

    for row in range(size):
        for col in range(size):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            cell = warped[y1:y2, x1:x2].copy()
            cells.append((cell, row, col))

    return cells


def center_crop(cell, margin_ratio=0.15):
    """Crop center of cell to remove grid lines.

    Args:
        cell: Cell image
        margin_ratio: Fraction of cell to remove from each edge

    Returns:
        Cropped cell image
    """
    h, w = cell.shape[:2]
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    return cell[margin_y:h-margin_y, margin_x:w-margin_x].copy()


def apply_augmentations(img, num_augmentations=5):
    """Apply random augmentations to increase dataset diversity.

    Args:
        img: Input image (grayscale)
        num_augmentations: Number of augmented versions to create

    Returns:
        List of augmented images (including original)
    """
    results = [img]  # Include original

    h, w = img.shape[:2]

    for _ in range(num_augmentations - 1):
        aug = img.copy()

        # Random rotation (-5 to +5 degrees)
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=255)

        # Random scale (0.9 to 1.1)
        scale = random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale), int(w * scale)
        aug = cv2.resize(aug, (new_w, new_h))

        # Pad or crop to original size
        if scale > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            aug = aug[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad with white
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            aug = cv2.copyMakeBorder(aug, pad_y, h-new_h-pad_y,
                                     pad_x, w-new_w-pad_x,
                                     cv2.BORDER_CONSTANT, value=255)

        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)

        # Random contrast adjustment
        contrast = random.uniform(0.8, 1.2)
        mean = np.mean(aug)
        aug = np.clip((aug - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise (occasionally)
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        results.append(aug)

    return results


def process_dataset(test_data_root, output_dir, augment=True, num_augmentations=5):
    """Process all test images and extract labeled cells.

    Args:
        test_data_root: Root directory containing test_data
        output_dir: Output directory for training cells
        augment: Whether to apply augmentations
        num_augmentations: Number of augmented versions per cell
    """
    # Create output directories for each digit (0-9)
    cells_dir = os.path.join(output_dir, 'cells')
    for digit in range(10):
        os.makedirs(os.path.join(cells_dir, str(digit)), exist_ok=True)

    # Collect all image-json pairs
    datasets = [
        ('classic_sudoku/9x9', 9),
        ('classic_sudoku/6x6', 6),
        ('killer_sudoku/9x9', 9),
        ('killer_sudoku/6x6', 6),
    ]

    all_images = []
    excluded_count = 0

    for subdir, size in datasets:
        dir_path = os.path.join(test_data_root, subdir)
        if os.path.exists(dir_path):
            # Get files to exclude for this directory
            exclude_for_dir = {
                f.split('/')[-1]
                for f in EXCLUDED_FILES
                if f.startswith(subdir)
            }
            if exclude_for_dir:
                print(f"  Excluding from {subdir}: {exclude_for_dir}")
                excluded_count += len(exclude_for_dir)

            for img_path, json_path in get_image_files(dir_path, exclude_for_dir):
                all_images.append((img_path, json_path, size))

    print(f"Excluded {excluded_count} files with known issues")

    print(f"Found {len(all_images)} images with ground truth")

    stats = {digit: 0 for digit in range(10)}
    cell_id = 0

    for img_path, json_path, size in all_images:
        print(f"Processing: {img_path}")

        # Load ground truth
        with open(json_path, 'r') as f:
            data = json.load(f)
        board = data['board']

        # Warp image
        warped = get_warped_grid(img_path)
        if warped is None:
            print(f"  WARNING: Could not warp {img_path}, skipping")
            continue

        # Extract cells
        cells = extract_cells(warped, size)

        for cell_img, row, col in cells:
            digit = board[row][col]

            # Convert to grayscale
            if len(cell_img.shape) == 3:
                cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            else:
                cell_gray = cell_img

            # Center crop to remove grid lines
            cell_cropped = center_crop(cell_gray, margin_ratio=0.12)

            # Resize to standard size (64x64 for CNN input)
            cell_resized = cv2.resize(cell_cropped, (64, 64))

            # Apply augmentations if enabled
            if augment:
                augmented = apply_augmentations(cell_resized, num_augmentations)
            else:
                augmented = [cell_resized]

            # Save each augmented version
            for i, aug_img in enumerate(augmented):
                filename = f"cell_{cell_id:05d}_aug{i}.png"
                save_path = os.path.join(cells_dir, str(digit), filename)
                cv2.imwrite(save_path, aug_img)
                stats[digit] += 1

            cell_id += 1

    print("\nDataset Statistics:")
    print("-" * 30)
    total = 0
    for digit in range(10):
        print(f"  Digit {digit}: {stats[digit]} samples")
        total += stats[digit]
    print(f"  TOTAL: {total} samples")

    # Save metadata
    metadata = {
        'total_samples': total,
        'samples_per_digit': stats,
        'augmentations_per_cell': num_augmentations if augment else 1,
        'image_size': 64,
        'source_images': len(all_images)
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to: {output_dir}")
    return metadata


def create_train_val_split(output_dir, val_ratio=0.2):
    """Create train/val split file lists.

    Args:
        output_dir: Directory containing cells/
        val_ratio: Fraction of data for validation
    """
    cells_dir = os.path.join(output_dir, 'cells')
    train_list = []
    val_list = []

    for digit in range(10):
        digit_dir = os.path.join(cells_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue

        files = [f for f in os.listdir(digit_dir) if f.endswith('.png')]
        random.shuffle(files)

        split_idx = int(len(files) * (1 - val_ratio))

        for f in files[:split_idx]:
            train_list.append(f"{digit}/{f}")
        for f in files[split_idx:]:
            val_list.append(f"{digit}/{f}")

    # Save split files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_list))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_list))

    print(f"\nTrain/Val split:")
    print(f"  Training samples: {len(train_list)}")
    print(f"  Validation samples: {len(val_list)}")


if __name__ == '__main__':
    # Paths
    script_dir = Path(__file__).parent
    test_data_root = script_dir.parent / 'test_data'
    output_dir = script_dir / 'training_data'

    print("Sudoku Cell Dataset Extractor")
    print("=" * 50)
    print(f"Test data root: {test_data_root}")
    print(f"Output directory: {output_dir}")
    print()

    # Process dataset with augmentations
    metadata = process_dataset(
        str(test_data_root),
        str(output_dir),
        augment=True,
        num_augmentations=5
    )

    # Create train/val split
    create_train_val_split(str(output_dir), val_ratio=0.2)

    print("\nDone!")
