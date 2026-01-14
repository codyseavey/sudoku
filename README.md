# Sudoku

Full-stack Sudoku application supporting classic and killer sudoku variants with image extraction (OCR) capabilities.

## Architecture

Three-service architecture:
- **Backend** (Go, port 8080) - API server for puzzle generation, solving, and statistics
- **Frontend** (Vue 3, port 5173 dev) - Web UI with 24+ solving strategies
- **Extraction Service** (Python/Flask, port 5001) - Image processing and OCR for puzzle extraction

```
Browser → Frontend (Vue 3)
           ↓
        Backend API (Go :8080)
           ├→ /api/puzzle     - Generate puzzle
           ├→ /api/solve      - Solve board
           ├→ /api/complete   - Track completed puzzles
           ├→ /api/upload     - Image upload
           │    └→ Extraction Service (Python :5001)
           └→ /api/stats      - Game statistics
```

## Prerequisites

- **Go** 1.24+
- **Node.js** 18+ and npm
- **Python** 3.10+
- **Google Cloud Vision credentials** (for OCR features) - store as `google-cloud-adminSvc.json` in repo root

## Quick Start

Use the development script to run all services:

```bash
# Run all services (backend, frontend, extraction)
./scripts/dev.sh

# Run individual services
./scripts/dev.sh backend
./scripts/dev.sh frontend
./scripts/dev.sh extraction
```

## Manual Setup

### Backend (Go)

```bash
cd backend
go build -o sudoku-server .
go run main.go              # Run server
go run main.go -verbose     # Run with verbose logging
```

### Frontend (Vue 3)

```bash
cd frontend
npm install
npm run dev     # Development server at http://localhost:5173
npm run build   # Production build to dist/
```

### Extraction Service (Python)

```bash
cd extraction_service
pip install -r requirements.txt

# Set Google Cloud credentials (required for OCR)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-cloud-adminSvc.json

python app.py --port 5001
```

## ML Model Training

Train the machine learning models used for digit and cage sum recognition:

```bash
# Train all models
./scripts/train-models.sh

# Train individual models
./scripts/train-models.sh digits      # Digit recognition CNN
./scripts/train-models.sh boundary    # Boundary detection classifier
./scripts/train-models.sh cage-sums   # Cage sum recognition CNN
```

### Manual Training Commands

```bash
cd extraction_service

# Digit CNN
python train_digit_cnn.py --epochs 20 --batch-size 32

# Boundary classifier
python extract_boundary_training_data.py
python train_boundary_classifier.py

# Cage sum CNN
python extract_cage_sum_training_data.py
python prepare_cage_sum_cnn_data.py
python train_cage_sum_cnn_v2.py --epochs 30 --batch-size 32
```

## Testing

### Backend Tests

```bash
cd backend
go test ./...                        # Run all tests
go test ./sudoku -run TestSolve      # Run specific test
```

### Extraction Validation

```bash
# Requires extraction service running
python test_data/test_classic_extraction.py
python test_data/test_killer_extraction.py
```

## Project Structure

```
├── backend/
│   ├── main.go              # Server entry point
│   └── sudoku/
│       ├── sudoku.go        # Core solver
│       ├── killer_*.go      # Killer sudoku logic
│       └── extraction.go    # Extraction service proxy
├── frontend/
│   └── src/
│       └── strategies/      # 24+ solving algorithms
├── extraction_service/
│   ├── app.py               # Flask service (4300+ lines)
│   ├── models/              # Trained ML models
│   │   ├── digit_cnn.pth
│   │   ├── cage_sum_cnn.pth
│   │   └── boundary_classifier_rf.pkl
│   └── train_*.py           # Model training scripts
├── test_data/
│   ├── classic_sudoku/      # Test images with ground truth
│   └── killer_sudoku/
├── scripts/
│   ├── dev.sh               # Local development script
│   └── train-models.sh      # Model training script
└── deployment/
    └── deploy.sh            # Production deployment
```

## Features

- **Classic Sudoku**: 6x6 and 9x9 grid sizes
- **Killer Sudoku**: Cage-based variant with sum constraints
- **Image Upload**: Extract puzzles from photos using OCR
- **Solving Strategies**: Naked/Hidden Singles, X-Wing, Swordfish, Y-Wing, Skyscraper, Unique Rectangle, and more
- **Unique Solution Validation**: Ensures puzzles have exactly one solution

## Key Constraints

- Puzzles must have exactly one unique solution
- Killer sudoku cage sums must total 405 (9x9) or 126 (6x6)
- Image upload limit: 10MB
- Grid warped to 1800x1800 for consistent OCR processing

## CI/CD

Push to `main` triggers GitHub Actions:
1. Backend tests (`go test ./...`)
2. Build Go binary and Vue frontend
3. Retrain ML models from test_data images
4. Deploy to production

Manual model retraining available via "Retrain ML Models" workflow.
