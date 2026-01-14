#!/bin/bash
# Local development script for Sudoku application
# Usage: ./scripts/dev.sh [service]
#   ./scripts/dev.sh           - Run all services
#   ./scripts/dev.sh backend   - Run backend only
#   ./scripts/dev.sh frontend  - Run frontend only
#   ./scripts/dev.sh extraction - Run extraction service only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

# Check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    local missing=0

    if ! check_command "go"; then
        missing=1
    fi

    if ! check_command "node"; then
        missing=1
    fi

    if ! check_command "npm"; then
        missing=1
    fi

    if ! check_command "python3"; then
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        log_error "Missing prerequisites. Please install the required tools."
        exit 1
    fi

    log_success "All prerequisites found"
}

# Run backend service
run_backend() {
    log_info "Starting backend service on port 8080..."
    cd "$PROJECT_ROOT/backend"
    go run main.go -verbose
}

# Run frontend service
run_frontend() {
    log_info "Starting frontend service on port 5173..."
    cd "$PROJECT_ROOT/frontend"

    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi

    npm run dev
}

# Run extraction service
run_extraction() {
    log_info "Starting extraction service on port 5001..."
    cd "$PROJECT_ROOT/extraction_service"

    # Check for Google Cloud credentials
    CREDS_FILE="$PROJECT_ROOT/google-cloud-adminSvc.json"
    if [ -f "$CREDS_FILE" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
        log_success "Google Cloud credentials loaded"
    else
        log_warn "Google Cloud credentials not found at $CREDS_FILE"
        log_warn "OCR features may not work without credentials"
    fi

    # Check if virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        log_info "Using virtual environment"
    else
        log_warn "No virtual environment found. Consider creating one with: python3 -m venv venv"
    fi

    # Install dependencies if needed
    if ! python3 -c "import flask" 2>/dev/null; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
    fi

    python3 app.py --port 5001
}

# Run all services in parallel
run_all() {
    log_info "Starting all services..."
    log_info "  Backend:    http://localhost:8080"
    log_info "  Frontend:   http://localhost:5173"
    log_info "  Extraction: http://localhost:5001"
    echo ""
    log_info "Press Ctrl+C to stop all services"
    echo ""

    # Trap to kill all background processes on exit
    trap 'kill $(jobs -p) 2>/dev/null; exit' INT TERM

    # Start services in background
    (cd "$PROJECT_ROOT/backend" && go run main.go -verbose) &
    BACKEND_PID=$!

    (
        cd "$PROJECT_ROOT/frontend"
        if [ ! -d "node_modules" ]; then
            npm install
        fi
        npm run dev
    ) &
    FRONTEND_PID=$!

    (
        cd "$PROJECT_ROOT/extraction_service"
        CREDS_FILE="$PROJECT_ROOT/google-cloud-adminSvc.json"
        if [ -f "$CREDS_FILE" ]; then
            export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
        fi
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        python3 app.py --port 5001
    ) &
    EXTRACTION_PID=$!

    # Wait for any process to exit
    wait
}

# Main
main() {
    cd "$PROJECT_ROOT"

    case "${1:-all}" in
        backend)
            check_command "go" || exit 1
            run_backend
            ;;
        frontend)
            check_command "node" || exit 1
            check_command "npm" || exit 1
            run_frontend
            ;;
        extraction)
            check_command "python3" || exit 1
            run_extraction
            ;;
        all)
            check_prerequisites
            run_all
            ;;
        -h|--help|help)
            echo "Usage: $0 [service]"
            echo ""
            echo "Services:"
            echo "  all         Run all services (default)"
            echo "  backend     Run backend only (Go, port 8080)"
            echo "  frontend    Run frontend only (Vue, port 5173)"
            echo "  extraction  Run extraction service only (Python, port 5001)"
            echo ""
            echo "Examples:"
            echo "  $0              # Run all services"
            echo "  $0 backend      # Run backend only"
            echo "  $0 frontend     # Run frontend only"
            ;;
        *)
            log_error "Unknown service: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
