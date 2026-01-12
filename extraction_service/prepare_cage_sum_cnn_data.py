#!/usr/bin/env python3
"""
Prepare cage sum crops for CNN training.
Extracts the top-left corner region where cage sums appear.

Key requirements:
- At least 2 samples of each label in training data
- At least 1 sample of each label in validation data
- Consistent preprocessing with warping/normalization
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import random

# Minimum samples required per split
MIN_TRAIN_SAMPLES = 2
MIN_VAL_SAMPLES = 1
TARGET_SAMPLES_PER_LABEL = 30  # Target for augmentation

def extract_cage_sum_region(cell_img, crop_h_ratio=0.40, crop_w_ratio=0.65):
    """Extract the top-left corner region where cage sum appears.

    Args:
        cell_img: Full cell image (200x200)
        crop_h_ratio: Height ratio to crop (0.40 = 40% of cell height)
        crop_w_ratio: Width ratio to crop (0.65 = 65% of cell width)

    Returns:
        Cropped region containing cage sum
    """
    h, w = cell_img.shape[:2]

    # Crop top-left corner
    crop_h = int(h * crop_h_ratio)
    crop_w = int(w * crop_w_ratio)

    # Add small margin from edges
    margin_top = 5
    margin_left = 8

    cropped = cell_img[margin_top:crop_h, margin_left:crop_w]

    return cropped


def normalize_crop(crop):
    """Apply consistent normalization to make crops more uniform.

    Args:
        crop: Input grayscale crop

    Returns:
        Normalized crop with consistent lighting/contrast
    """
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization for consistent contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(crop)

    # Light denoising while preserving edges
    normalized = cv2.bilateralFilter(normalized, 5, 50, 50)

    return normalized


def deskew_crop(crop):
    """Deskew a crop to correct for rotation.

    Args:
        crop: Input grayscale crop

    Returns:
        Deskewed crop
    """
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Find moments to determine skew angle
    coords = np.column_stack(np.where(crop < 200))
    if len(coords) < 10:
        return crop

    # Use PCA-like approach via moments
    moments = cv2.moments(255 - crop)
    if moments['mu02'] == 0:
        return crop

    # Calculate skew angle
    skew = moments['mu11'] / moments['mu02']
    angle = np.degrees(np.arctan(skew))

    # Limit correction to small angles
    if abs(angle) > 15:
        return crop

    # Apply rotation to deskew
    h, w = crop.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(crop, M, (w, h), borderValue=255)

    return deskewed

def preprocess_for_cnn(crop, target_size=(64, 64), apply_deskew=True, apply_normalize=True):
    """Preprocess crop for CNN input with consistent normalization.

    Args:
        crop: Cage sum crop
        target_size: Target size for CNN (64x64)
        apply_deskew: Whether to apply deskewing
        apply_normalize: Whether to apply normalization

    Returns:
        Preprocessed image ready for CNN
    """
    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Apply deskewing for consistency
    if apply_deskew:
        gray = deskew_crop(gray)

    # Apply normalization for consistent contrast/lighting
    if apply_normalize:
        gray = normalize_crop(gray)

    # Resize to target size with high-quality interpolation
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Final CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return enhanced

def augment_crop(crop, target_count=30):
    """Create augmented versions of a crop with enhanced variations.

    Includes warping transforms for better generalization.

    Args:
        crop: Input crop to augment
        target_count: Target number of augmented samples to generate

    Returns:
        List of augmented crops (exactly target_count samples)
    """
    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = crop.shape
    augmented = [crop.copy()]  # Original

    # Rotation variations (more angles for better coverage)
    for angle in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (w, h), borderValue=255)
        augmented.append(rotated)

    # Scaling variations
    for scale in [0.88, 0.92, 0.96, 1.04, 1.08, 1.12]:
        scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Crop/pad to original size
        if scaled.shape[0] > h:
            start = (scaled.shape[0] - h) // 2
            scaled = scaled[start:start+h, :]
        elif scaled.shape[0] < h:
            pad = (h - scaled.shape[0]) // 2
            scaled = cv2.copyMakeBorder(scaled, pad, h-scaled.shape[0]-pad, 0, 0,
                                       cv2.BORDER_CONSTANT, value=255)
        if scaled.shape[1] > w:
            start = (scaled.shape[1] - w) // 2
            scaled = scaled[:, start:start+w]
        elif scaled.shape[1] < w:
            pad = (w - scaled.shape[1]) // 2
            scaled = cv2.copyMakeBorder(scaled, 0, 0, pad, w-scaled.shape[1]-pad,
                                       cv2.BORDER_CONSTANT, value=255)
        augmented.append(scaled)

    # Brightness variations
    for beta in [-30, -20, -10, 10, 20, 30]:
        adjusted = cv2.convertScaleAbs(crop, alpha=1.0, beta=beta)
        augmented.append(adjusted)

    # Contrast variations
    for alpha in [0.8, 0.9, 1.1, 1.2]:
        adjusted = cv2.convertScaleAbs(crop, alpha=alpha, beta=0)
        augmented.append(adjusted)

    # Gaussian noise
    for sigma in [3, 5, 8, 12]:
        noisy = crop.copy().astype(np.float32)
        noise = np.random.normal(0, sigma, crop.shape)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)

    # Gaussian blur (simulate focus issues)
    for ksize in [3, 5]:
        blurred = cv2.GaussianBlur(crop, (ksize, ksize), 0)
        augmented.append(blurred)

    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(crop, -1, kernel)
    augmented.append(sharpened)

    # Perspective transforms (warping for consistency training)
    # Horizontal skew
    for skew in [0.03, 0.06, -0.03, -0.06]:
        pts1 = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
        pts2 = np.array([[0,0], [w,0], [int(w*skew),h], [w-int(w*skew),h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(crop, M, (w, h), borderValue=255)
        augmented.append(warped)

    # Vertical skew
    for skew in [0.03, 0.06, -0.03, -0.06]:
        pts1 = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
        pts2 = np.array([[int(h*skew),0], [w-int(h*skew),0], [0,h], [w,h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(crop, M, (w, h), borderValue=255)
        augmented.append(warped)

    # Corner perspective warps (simulate camera angle)
    for dx, dy in [(3, 3), (-3, -3), (3, -3), (-3, 3), (5, 2), (-5, -2)]:
        pts1 = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
        pts2 = np.array([[dx,dy], [w-dx,dy], [0,h], [w,h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(crop, M, (w, h), borderValue=255)
        augmented.append(warped)

    # Affine transforms (shear)
    for shear in [0.05, -0.05, 0.08, -0.08]:
        M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(crop, M, (w, h), borderValue=255)
        augmented.append(sheared)

    # Elastic distortion (subtle)
    for strength in [2, 4]:
        distorted = elastic_distortion(crop, alpha=strength, sigma=3)
        augmented.append(distorted)

    # Combinations of transforms: Rotation + brightness
    for angle, beta in [(-3, 15), (3, -15), (-2, 25), (2, -25), (-4, 10), (4, -10)]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (w, h), borderValue=255)
        combined = cv2.convertScaleAbs(rotated, alpha=1.0, beta=beta)
        augmented.append(combined)

    # Return exactly target_count samples (randomly sample if too many)
    if len(augmented) > target_count:
        # Always include original
        indices = [0] + list(np.random.choice(range(1, len(augmented)), target_count - 1, replace=False))
        return [augmented[i] for i in indices]
    elif len(augmented) < target_count:
        # Repeat augmentations to reach target
        while len(augmented) < target_count:
            augmented.append(augmented[np.random.randint(1, len(augmented))])

    return augmented


def elastic_distortion(image, alpha=2, sigma=3):
    """Apply elastic distortion to image.

    Args:
        image: Input image
        alpha: Distortion strength
        sigma: Smoothness of distortion

    Returns:
        Distorted image
    """
    h, w = image.shape[:2]

    # Generate random displacement fields
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Apply remapping
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)

    return distorted

def main():
    """Prepare cage sum training data from extracted crops."""

    input_dir = Path(__file__).parent / 'training_data' / 'cage_sums'
    output_dir = Path(__file__).parent / 'training_data' / 'cage_sum_cnn'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing cage sum crops from: {input_dir}")
    print(f"Output to: {output_dir}")
    print("=" * 80)

    # Load all crops
    crop_files = sorted(input_dir.glob('*.png'))
    crop_files = [f for f in crop_files if f.name != 'extraction_metadata.json']

    print(f"Found {len(crop_files)} cage sum crops")

    # Organize by label
    data_by_label = {}  # label -> list of crops

    for crop_file in crop_files:
        # Parse filename: puzzle_cage_cageid_sum_VALUE_rX_cY.png
        parts = crop_file.stem.split('_')

        # Find sum value
        sum_idx = parts.index('sum') if 'sum' in parts else -1
        if sum_idx == -1 or sum_idx + 1 >= len(parts):
            print(f"Warning: Could not parse sum from {crop_file.name}")
            continue

        sum_value = int(parts[sum_idx + 1])

        # Load and process crop
        cell_img = cv2.imread(str(crop_file))
        if cell_img is None:
            print(f"Warning: Could not load {crop_file.name}")
            continue

        # Extract cage sum region
        cage_sum_crop = extract_cage_sum_region(cell_img)

        if cage_sum_crop.size == 0:
            print(f"Warning: Empty crop for {crop_file.name}")
            continue

        # Preprocess
        processed = preprocess_for_cnn(cage_sum_crop)

        # Store
        if sum_value not in data_by_label:
            data_by_label[sum_value] = []

        data_by_label[sum_value].append({
            'image': processed,
            'source_file': crop_file.name
        })

    # Print statistics
    print("\nCrops per label:")
    for label in sorted(data_by_label.keys()):
        count = len(data_by_label[label])
        print(f"  Label {label:2d}: {count:3d} crops")

    # Augment ALL labels with <30 samples to improve class balance
    target_count = 30  # Target at least 30 samples per label
    augmented_labels = []

    print("\nAugmenting rare labels...")
    for label in sorted(data_by_label.keys()):
        current_count = len(data_by_label[label])

        if current_count >= target_count:
            print(f"  Label {label:2d}: {current_count:3d} crops (sufficient)")
            continue

        print(f"  Label {label:2d}: {current_count:3d} crops -> augmenting to {target_count}")

        # Augment existing crops
        original_crops = data_by_label[label].copy()
        augmented_data = []

        # For each original crop, create multiple augmented versions
        for crop_data in original_crops:
            # Get original image before preprocessing
            source_file = input_dir / crop_data['source_file']
            cell_img = cv2.imread(str(source_file))
            cage_sum_crop = extract_cage_sum_region(cell_img)

            # Calculate how many augmentations needed per crop
            needed = target_count - current_count
            augs_per_crop = max(1, needed // current_count + 1)

            # Create augmented versions
            augmented = augment_crop(cage_sum_crop, target_count=augs_per_crop)

            for aug_crop in augmented[1:]:  # Skip original (already in original_crops)
                processed = preprocess_for_cnn(aug_crop)
                augmented_data.append({
                    'image': processed,
                    'source_file': f"{crop_data['source_file']}_aug"
                })

        # Combine original and augmented, randomly sample to reach target count
        all_samples = original_crops + augmented_data
        random.shuffle(all_samples)
        data_by_label[label] = all_samples[:target_count]

        augmented_labels.append(label)
        print(f"    Result: {len(data_by_label[label])} crops")

    # Save processed crops
    print("\nSaving processed crops...")
    saved_count = 0

    for label, crops in data_by_label.items():
        label_dir = output_dir / f"label_{label:02d}"
        label_dir.mkdir(exist_ok=True)

        for idx, crop_data in enumerate(crops):
            filename = f"{label:02d}_{idx:04d}.png"
            filepath = label_dir / filename
            cv2.imwrite(str(filepath), crop_data['image'])
            saved_count += 1

    print(f"Saved {saved_count} processed crops to {output_dir}")

    # Create train/val split with guaranteed minimum samples per label
    print(f"\nCreating train/val split (ensuring min {MIN_TRAIN_SAMPLES} train, {MIN_VAL_SAMPLES} val per label)...")
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'

    # Clear existing directories
    import shutil
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    train_count = 0
    val_count = 0
    insufficient_labels = []

    for label, crops in data_by_label.items():
        # Shuffle for randomness
        random.seed(42 + label)  # Seed per label for reproducibility
        shuffled = crops.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        min_required = MIN_TRAIN_SAMPLES + MIN_VAL_SAMPLES

        if total < min_required:
            print(f"  Warning: Label {label} has only {total} samples (need {min_required})")
            insufficient_labels.append(label)
            # Split what we have, prioritizing training
            if total >= MIN_TRAIN_SAMPLES:
                train_crops = shuffled[:MIN_TRAIN_SAMPLES]
                val_crops = shuffled[MIN_TRAIN_SAMPLES:]
            else:
                train_crops = shuffled
                val_crops = []
        else:
            # Guarantee minimum val samples first
            val_crops = shuffled[:MIN_VAL_SAMPLES]
            remaining = shuffled[MIN_VAL_SAMPLES:]

            # Guarantee minimum train samples
            train_crops = remaining[:MIN_TRAIN_SAMPLES]
            remaining = remaining[MIN_TRAIN_SAMPLES:]

            # Split remaining 80/20
            if remaining:
                extra_train = int(len(remaining) * 0.8)
                train_crops.extend(remaining[:extra_train])
                val_crops.extend(remaining[extra_train:])

        # Save train
        if train_crops:
            train_label_dir = train_dir / f"label_{label:02d}"
            train_label_dir.mkdir(exist_ok=True)

            for idx, crop_data in enumerate(train_crops):
                filename = f"{label:02d}_{idx:04d}.png"
                cv2.imwrite(str(train_label_dir / filename), crop_data['image'])
                train_count += 1

        # Save val
        if val_crops:
            val_label_dir = val_dir / f"label_{label:02d}"
            val_label_dir.mkdir(exist_ok=True)

            for idx, crop_data in enumerate(val_crops):
                filename = f"{label:02d}_{idx:04d}.png"
                cv2.imwrite(str(val_label_dir / filename), crop_data['image'])
                val_count += 1

    print(f"Train: {train_count} samples")
    print(f"Val: {val_count} samples")

    if insufficient_labels:
        print(f"\n⚠️  Labels with insufficient samples: {insufficient_labels}")
        print(f"   These labels need at least {MIN_TRAIN_SAMPLES + MIN_VAL_SAMPLES} samples total")

    # Count samples per label in train/val
    train_per_label = {}
    val_per_label = {}
    for label in data_by_label.keys():
        train_label_dir = train_dir / f"label_{label:02d}"
        val_label_dir = val_dir / f"label_{label:02d}"
        train_per_label[label] = len(list(train_label_dir.glob('*.png'))) if train_label_dir.exists() else 0
        val_per_label[label] = len(list(val_label_dir.glob('*.png'))) if val_label_dir.exists() else 0

    # Verify minimum requirements
    print("\nVerifying minimum sample requirements...")
    all_ok = True
    for label in sorted(data_by_label.keys()):
        t_count = train_per_label.get(label, 0)
        v_count = val_per_label.get(label, 0)
        if t_count < MIN_TRAIN_SAMPLES or v_count < MIN_VAL_SAMPLES:
            print(f"  ❌ Label {label:2d}: train={t_count}, val={v_count}")
            all_ok = False
        else:
            print(f"  ✓  Label {label:2d}: train={t_count}, val={v_count}")

    if all_ok:
        print("\n✅ All labels meet minimum requirements!")
    else:
        print("\n⚠️  Some labels do not meet minimum requirements!")

    # Save metadata
    metadata = {
        'total_crops': saved_count,
        'num_classes': len(data_by_label),
        'train_samples': train_count,
        'val_samples': val_count,
        'min_train_per_label': MIN_TRAIN_SAMPLES,
        'min_val_per_label': MIN_VAL_SAMPLES,
        'samples_per_label': {k: len(v) for k, v in data_by_label.items()},
        'train_per_label': train_per_label,
        'val_per_label': val_per_label,
        'augmented_labels': augmented_labels,
        'insufficient_labels': insufficient_labels,
        'target_size': [64, 64]
    }

    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

if __name__ == '__main__':
    main()
