#!/bin/bash
# ML Model training script for Sudoku extraction service
# Usage: ./scripts/train-models.sh [model]
#   ./scripts/train-models.sh           - Train all models
#   ./scripts/train-models.sh digits    - Train digit recognition CNN
#   ./scripts/train-models.sh boundary  - Train boundary classifier
#   ./scripts/train-models.sh cage-sums - Train cage sum CNN

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTRACTION_DIR="$PROJECT_ROOT/extraction_service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python and dependencies
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    cd "$EXTRACTION_DIR"

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        log_info "Using virtual environment"
    fi

    # Check for required packages
    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch is not installed. Run: pip install -r requirements.txt"
        exit 1
    fi

    log_success "Python environment ready"
}

# Train digit recognition CNN
train_digits() {
    log_info "Training digit recognition CNN..."
    cd "$EXTRACTION_DIR"

    # Extract training data if needed
    if [ ! -d "training_data/digits" ] || [ -z "$(ls -A training_data/digits 2>/dev/null)" ]; then
        log_info "Extracting digit training data from test images..."
        python3 extract_training_cells.py
    fi

    python3 train_digit_cnn.py --epochs 20 --batch-size 32

    if [ -f "models/digit_cnn.pth" ]; then
        log_success "Digit CNN trained successfully: models/digit_cnn.pth"
    else
        log_error "Failed to train digit CNN"
        exit 1
    fi
}

# Train boundary classifier
train_boundary() {
    log_info "Training boundary classifier..."
    cd "$EXTRACTION_DIR"

    log_info "Extracting boundary training data..."
    python3 extract_boundary_training_data.py

    log_info "Training Random Forest classifier..."
    python3 train_boundary_classifier.py

    if [ -f "models/boundary_classifier_rf.pkl" ]; then
        log_success "Boundary classifier trained successfully: models/boundary_classifier_rf.pkl"
    else
        log_error "Failed to train boundary classifier"
        exit 1
    fi
}

# Train cage sum CNN
train_cage_sums() {
    log_info "Training cage sum recognition CNN..."
    cd "$EXTRACTION_DIR"

    log_info "Extracting cage sum training data..."
    python3 extract_cage_sum_training_data.py

    log_info "Preparing CNN training data..."
    python3 prepare_cage_sum_cnn_data.py

    log_info "Training cage sum CNN..."
    if [ -f "train_cage_sum_cnn_v2.py" ]; then
        python3 train_cage_sum_cnn_v2.py --epochs 30 --batch-size 32
    else
        python3 train_cage_sum_cnn.py --epochs 30 --batch-size 32
    fi

    if [ -f "models/cage_sum_cnn.pth" ]; then
        log_success "Cage sum CNN trained successfully: models/cage_sum_cnn.pth"
    else
        log_error "Failed to train cage sum CNN"
        exit 1
    fi
}

# Train all models
train_all() {
    log_info "Training all ML models..."
    echo ""

    train_digits
    echo ""

    train_boundary
    echo ""

    train_cage_sums
    echo ""

    log_success "All models trained successfully!"
    echo ""
    log_info "Models saved to: $EXTRACTION_DIR/models/"
    ls -la "$EXTRACTION_DIR/models/"
}

# Validate models
validate_models() {
    log_info "Validating trained models..."
    cd "$PROJECT_ROOT"

    echo ""
    log_info "Testing classic sudoku extraction..."
    python3 test_data/test_classic_extraction.py

    echo ""
    log_info "Testing killer sudoku extraction..."
    python3 test_data/test_killer_extraction.py
}

# Main
main() {
    case "${1:-all}" in
        digits|digit)
            check_python
            train_digits
            ;;
        boundary|boundaries)
            check_python
            train_boundary
            ;;
        cage-sums|cage-sum|cagesums|cagesum)
            check_python
            train_cage_sums
            ;;
        all)
            check_python
            train_all
            ;;
        validate|test)
            check_python
            validate_models
            ;;
        -h|--help|help)
            echo "Usage: $0 [model]"
            echo ""
            echo "Models:"
            echo "  all         Train all models (default)"
            echo "  digits      Train digit recognition CNN"
            echo "  boundary    Train boundary detection classifier"
            echo "  cage-sums   Train cage sum recognition CNN"
            echo "  validate    Run extraction validation tests"
            echo ""
            echo "Output:"
            echo "  models/digit_cnn.pth           - Digit recognition (PyTorch)"
            echo "  models/cage_sum_cnn.pth        - Cage sum recognition (PyTorch)"
            echo "  models/boundary_classifier_rf.pkl - Boundary detection (sklearn)"
            echo ""
            echo "Examples:"
            echo "  $0              # Train all models"
            echo "  $0 digits       # Train digit CNN only"
            echo "  $0 validate     # Test extraction accuracy"
            ;;
        *)
            log_error "Unknown model: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
