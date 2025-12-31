package sudoku

import (
	"testing"
)

func TestGenerate9x9(t *testing.T) {
	puzzle := Generate("medium", 9)

	if len(puzzle.Board) != 9 || len(puzzle.Board[0]) != 9 {
		t.Errorf("Expected 9x9 board, got %dx%d", len(puzzle.Board), len(puzzle.Board[0]))
	}

	// Verify solution is valid
	if !isValidGrid(puzzle.Solution, 9, 3, 3) {
		t.Error("Generated solution is invalid")
	}

	// Verify board matches solution (where filled)
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if puzzle.Board[i][j] != 0 && puzzle.Board[i][j] != puzzle.Solution[i][j] {
				t.Errorf("Board mismatch at (%d,%d)", i, j)
			}
		}
	}
}

func TestGenerate6x6(t *testing.T) {
	puzzle := Generate("medium", 6)

	if len(puzzle.Board) != 6 || len(puzzle.Board[0]) != 6 {
		t.Errorf("Expected 6x6 board, got %dx%d", len(puzzle.Board), len(puzzle.Board[0]))
	}

	// Verify solution is valid (2x3 boxes)
	if !isValidGrid(puzzle.Solution, 6, 2, 3) {
		t.Error("Generated 6x6 solution is invalid")
	}
}

func TestSolveUnique(t *testing.T) {
	// Simple easy 4x4 or 9x9 puzzle known to be unique
	// Let's use a 9x9 generated one with few holes
	puzzle := Generate("easy", 9)
	
	// Solve it
	sol, ok := Solve(puzzle.Board, 9)
	if !ok {
		t.Error("Failed to solve a valid generated puzzle")
	}
	
	// Verify solution matches
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if sol[i][j] != puzzle.Solution[i][j] {
				t.Errorf("Solved grid does not match expected solution at (%d,%d)", i, j)
			}
		}
	}
}

func TestSolveInvalid(t *testing.T) {
	// Create a grid with a conflict
	g := make(Grid, 9)
	for i := range g {
		g[i] = make([]int, 9)
	}
	g[0][0] = 1
	g[0][1] = 1 // Conflict in row

	_, ok := Solve(g, 9)
	if ok {
		t.Error("Solve should fail for invalid grid")
	}
}

func TestSolveMultipleSolutions(t *testing.T) {
	// Empty grid has multiple solutions
	g := make(Grid, 9)
	for i := range g {
		g[i] = make([]int, 9)
	}

	_, ok := Solve(g, 9)
	if ok {
		t.Error("Solve should fail (return false) for grid with multiple solutions")
	}
}

// Helper to check if a full grid is valid
func isValidGrid(g Grid, n, boxH, boxW int) bool {
	gen := Generator{N: n, BoxHeight: boxH, BoxWidth: boxW}
	
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val := g[i][j]
			if val == 0 {
				return false // Not full
			}
			
			// Check checks
			// We need to temporarily zero the cell to use isSafe
			g[i][j] = 0
			safe := gen.isSafe(g, i, j, val)
			g[i][j] = val
			
			if !safe {
				return false
			}
		}
	}
	return true
}
