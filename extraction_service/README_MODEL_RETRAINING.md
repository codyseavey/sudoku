# ML Model Retraining Guide

This guide explains how to retrain the ML models used in the Sudoku extraction service.

## Models

The system uses two ML models:

1. **Digit CNN** (`digit_cnn.pth`, 8.6 MB)
   - LeNet-5 inspired convolutional neural network
   - Recognizes digits 0-9 in sudoku cells
   - Trained on 3,000+ cell images from real puzzles

2. **Boundary Classifier** (`boundary_classifier_rf.pkl`, 470 KB)
   - Random Forest classifier (100 estimators)
   - Detects cage boundaries in killer sudoku
   - Uses 38-dimensional feature vectors (edge detection, FFT, morphology)

## Automated Retraining via GitHub Actions

### Manual Trigger (Recommended)

1. Go to **Actions** tab in GitHub
2. Select **"Retrain ML Models"** workflow
3. Click **"Run workflow"**
4. Choose which model(s) to retrain:
   - `digit-cnn` - Retrain only the digit classifier
   - `boundary-classifier` - Retrain only the boundary detector
   - `both` - Retrain all models

The workflow will:
1. ✅ Train selected models on the latest training data
2. ✅ Automatically commit retrained models to `main` branch
3. ✅ Push commit → triggers CI/CD deployment pipeline
4. ✅ Models automatically deployed to production server

**Total time:** ~5-10 minutes (training + deployment)

### Scheduled Retraining (Optional)

Uncomment the schedule in `.github/workflows/retrain-models.yml`:

```yaml
schedule:
  - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM UTC
```

This enables automatic weekly retraining to incorporate new training data.

## Training Data

### Digit CNN Training Data

Location: `extraction_service/training_data/cells/`

Structure:
```
training_data/
├── cells/
│   ├── 0/  # Empty cell images
│   ├── 1/  # Digit 1 images
│   ├── 2/  # Digit 2 images
│   ...
│   └── 9/  # Digit 9 images
├── train.txt  # Training split file list
├── val.txt    # Validation split file list
└── metadata.json
```

**Total:** ~3,000 images (61 MB)

### Boundary Classifier Training Data

Location: `extraction_service/training_data/boundary_crops/`

Structure:
```
training_data/
├── boundary_crops/
│   ├── positive/  # True cage boundaries
│   └── negative/  # False positives (grid lines, etc.)
├── boundary_features.npz  # Precomputed features
└── false_negatives/  # Missed boundaries for debugging
```

**Total:** ~800 boundary crop images (28 MB)

## Local Retraining (Development)

### Prerequisites

```bash
cd extraction_service
source venv/bin/activate  # or activate your virtualenv
pip install -r requirements.txt
```

### Retrain Digit CNN

```bash
python train_digit_cnn.py --epochs 20 --batch-size 32 --lr 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: cuda or cpu (default: auto-detect)

Output: `models/digit_cnn.pth`

### Retrain Boundary Classifier

```bash
# Extract features from boundary crops
python extract_boundary_training_data.py

# Train Random Forest classifier
python train_boundary_classifier.py
```

Output:
- `models/boundary_classifier_rf.pkl`
- `models/boundary_scaler.pkl`

### Test Retrained Models

After retraining, test with the killer sudoku test suite:

```bash
cd ../test_data/killer_sudoku
python test_all.py
```

Expected: 19+/21 PASS (90%+)

## Adding New Training Data

### Adding Digit Training Data

1. Extract cells from new puzzles:
   ```bash
   python extract_training_cells.py --puzzle-dir path/to/puzzles
   ```

2. Manually verify and organize extracted cells into `training_data/cells/0-9/`

3. Update `train.txt` and `val.txt` with new file paths

4. Retrain: `python train_digit_cnn.py`

### Adding Boundary Training Data

1. Run boundary extraction on new puzzles:
   ```bash
   python extract_boundary_training_data.py --puzzles path/to/new/puzzles
   ```

2. Review false positives/negatives in `training_data/false_negatives/`

3. Move corrected samples to `boundary_crops/positive/` or `negative/`

4. Retrain: `python train_boundary_classifier.py`

## Model Performance

### Current Baselines (as of Jan 2026)

**Digit CNN:**
- Training accuracy: ~99%
- Validation accuracy: ~98%
- Real-world performance: Successfully reads placed digits in 95%+ of cells

**Boundary Classifier:**
- Test accuracy: 100% on validation set
- Real-world performance: 90% PASS rate (19/21 puzzles)
- Precision: High (few false cage boundaries)
- Recall: Good (detects most boundaries)

## Troubleshooting

### Model not deploying after retraining

Check GitHub Actions logs:
1. Verify "Commit and push retrained models" step succeeded
2. Check CI/CD pipeline was triggered by the commit
3. Verify "Deploy ML models" step in deployment

### Training fails with CUDA errors

Train on CPU instead:
```bash
python train_digit_cnn.py --device cpu
```

### Low accuracy after retraining

- Check training data quality (corrupted images?)
- Verify train/val split (enough validation data?)
- Try different hyperparameters (learning rate, epochs)

## Architecture Details

### Digit CNN (LeNet-5 inspired)

```
Input: [1, 64, 64] grayscale image
├─ Conv2d(1, 32, 5x5) + ReLU + MaxPool(2x2)
├─ Conv2d(32, 64, 5x5) + ReLU + MaxPool(2x2)
├─ Conv2d(64, 128, 3x3) + ReLU + MaxPool(2x2)
├─ Flatten
├─ Linear(6272 → 256) + ReLU + Dropout(0.5)
├─ Linear(256 → 128) + ReLU + Dropout(0.5)
└─ Linear(128 → 10)  # 10 classes (0-9)
```

### Boundary Classifier (Random Forest)

```
Input: 38-dimensional feature vector
├─ Edge detection features (12D)
│  ├─ Canny edge density
│  ├─ Sobel gradient stats
│  └─ Edge orientation histogram
├─ Frequency features (8D)
│  ├─ FFT magnitude
│  └─ Dominant frequencies
├─ Morphological features (10D)
│  ├─ Opening/closing responses
│  └─ Line detection (vertical/horizontal)
└─ Texture features (8D)
   ├─ Standard deviation
   └─ Contrast metrics

Random Forest (100 estimators, max_depth=20)
└─ Output: boundary probability [0, 1]
```

## CI/CD Integration

The retraining workflow integrates seamlessly with deployment:

```
Manual Trigger
    ↓
Train Models (GitHub Actions)
    ↓
Commit to main branch
    ↓
Push → Triggers CI/CD
    ↓
Deploy to Production (192.168.86.227)
    ↓
Verify ML models loaded
    ↓
✅ Production ready with new models
```

**Total automation time:** ~10 minutes from retrain trigger to production deployment

## Best Practices

1. **Always test locally first** before committing training data changes
2. **Retrain both models together** (`both` option) when adding new puzzle types
3. **Monitor test suite results** after retraining (should maintain 90%+ PASS)
4. **Keep training data quality high** - manually verify extracted samples
5. **Version models** - GitHub automatically creates artifacts for rollback

## Contact

For issues with model retraining, check:
- GitHub Actions logs in Actions tab
- Production service logs: `ssh 192.168.86.227 'sudo journalctl -u sudoku-extraction'`
- Test suite results: `test_data/killer_sudoku/test_all.py`
