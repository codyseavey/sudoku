package sudoku

import (
	"testing"
)

func TestGenerateKiller(t *testing.T) {
	puzzle := GenerateKiller("medium", 9)

	if puzzle.GameType != "killer" {
		t.Errorf("Expected GameType to be 'killer', got %s", puzzle.GameType)
	}

	if len(puzzle.Cages) == 0 {
		t.Error("Expected cages to be generated, got 0")
	}

	// Verify all cells are covered exactly once
	covered := make([][]bool, 9)
	for i := range covered {
		covered[i] = make([]bool, 9)
	}

	totalCells := 0
	for _, cage := range puzzle.Cages {
		currentSum := 0
		values := make(map[int]bool)
		for _, cell := range cage.Cells {
			if covered[cell.Row][cell.Col] {
				t.Errorf("Cell (%d, %d) is covered by multiple cages", cell.Row, cell.Col)
			}
			covered[cell.Row][cell.Col] = true
			totalCells++

			val := puzzle.Solution[cell.Row][cell.Col]
			currentSum += val
			if values[val] {
				t.Errorf("Duplicate value %d in cage at (%d, %d)", val, cell.Row, cell.Col)
			}
			values[val] = true
		}
		if currentSum != cage.Sum {
			t.Errorf("Cage sum mismatch: expected %d, got %d", cage.Sum, currentSum)
		}
	}

	if totalCells != 81 {
		t.Errorf("Expected 81 cells covered, got %d", totalCells)
	}

	// Verify Board is empty
	for i:=0; i<9; i++ {
		for j:=0; j<9; j++ {
			if puzzle.Board[i][j] != 0 {
				t.Errorf("Expected board to be empty, but found %d at (%d,%d)", puzzle.Board[i][j], i, j)
			}
		}
	}
}

func TestGenerateKiller6x6(t *testing.T) {
	puzzle := GenerateKiller("medium", 6)

	if len(puzzle.Cages) == 0 {
		t.Error("Expected cages to be generated, got 0")
	}

	totalCells := 0
	for _, cage := range puzzle.Cages {
		for range cage.Cells {
			totalCells++
		}
	}

	if totalCells != 36 {
		t.Errorf("Expected 36 cells covered, got %d", totalCells)
	}
}
