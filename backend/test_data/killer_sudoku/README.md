# Killer Sudoku Extraction Scripts

This directory contains test data for Killer Sudoku puzzles.

## CV Extraction Script

A Computer Vision based extraction script is located at `scripts/extract_killer_sudoku_cv.py`.
This script attempts to extract the puzzle definition (grid, cages, sums) from the image without using an LLM.

### Requirements

*   Python 3
*   OpenCV (`opencv-python-headless`)
*   NumPy
*   NetworkX
*   Scikit-image (optional)

### Usage

```bash
python3 scripts/extract_killer_sudoku_cv.py backend/test_data/killer_sudoku backend/test_data/killer_sudoku/1.png backend/test_data/killer_sudoku/1.json
```

### Notes

*   The script uses `1.png` and `1.json` to "learn" the digit templates (0-9).
*   It then attempts to process other images in the directory.
*   **Verification:** It checks if the sum of all cage sums equals 405. If not, it warns and does not save the JSON to avoid generating incorrect data.
*   **Current Status:** The boundary detection for dashed lines is sensitive to image quality and scaling. You may need to tune the `check_boundary` threshold in the script for different image sets.
