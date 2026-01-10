#!/bin/bash
set -e

echo "Starting codebase cleanup..."
echo "This will remove debug scripts and artifacts (52 scripts + 41 images ~12MB)"
echo ""

# Delete killer sudoku debug scripts
cd test_data/killer_sudoku
echo "Removing debug scripts from test_data/killer_sudoku/..."
rm -f analyze_*.py check_*.py debug_*.py compare_*.py examine_*.py
rm -f find_*.py save_*.py trace_*.py visualize_*.py
rm -rf debug_crops_13/

# Delete debug images
echo "Removing debug images from test_data/killer_sudoku/..."
rm -f boundary_*.png cell_*.png crop_*.png
rm -f debug_*.png false_pos_*.png missing_*.png
rm -f ocr_crop_*.png strip_*.png full_cell_*.png
rm -f cleaned_*.png warped_*.png extended_crop_*.png
rm -f fp_*.png large_crop_*.png

# Delete test_data root utilities
cd ..
echo "Removing temporary test utilities..."
rm -f format_json.py generate_training_data.py

# Delete extraction service debugging scripts
cd ../extraction_service
echo "Removing extraction service debug scripts..."
rm -f extract_misclassified_boundaries.py

# Delete duplicate model
cd models
if [ -f digit_cnn_final.pth ]; then
    echo "Removing duplicate model file..."
    rm -f digit_cnn_final.pth
fi

echo ""
echo "Cleanup complete!"
echo "Removed ~52 debug scripts and 41 debug images"
echo ""
echo "Retained production files:"
echo "  - test_data/test_classic_ocr.py"
echo "  - test_data/validate_*.py"
echo "  - test_data/killer_sudoku/test_all.py"
echo "  - extraction_service/train_*.py"
echo "  - extraction_service/models/*.{pth,pkl}"
