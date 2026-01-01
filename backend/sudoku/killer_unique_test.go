package sudoku

import (
	"testing"
)

func TestGenerateKillerUniqueness(t *testing.T) {
	size := 6 // Use 6x6 for speed
	// Test Hard (empty board, pure cage constraints)
	pHard := GenerateKiller("hard", size)

	// Verify it has exactly one solution
	// We need to construct a generator to call solveCountKiller
	gen := Generator{
		N: size,
		BoxHeight: 2,
		BoxWidth: 3,
		Cages: pHard.Cages,
	}

	// Start with empty board
	emptyBoard := make(Grid, size)
	for i := range emptyBoard {
		emptyBoard[i] = make([]int, size)
	}

	count := 0
	gen.solveCountKiller(emptyBoard, &count)

	if count != 1 {
		t.Errorf("GenerateKiller('hard') produced a puzzle with %d solutions", count)
	}

	// Test Easy (should also be unique, but logic is different)
	pEasy := GenerateKiller("easy", size)
	genEasy := Generator{
		N: size,
		BoxHeight: 2,
		BoxWidth: 3,
		Cages: pEasy.Cages,
	}

	// Copy board
	boardEasy := make(Grid, size)
	for i := range pEasy.Board {
		boardEasy[i] = make([]int, size)
		copy(boardEasy[i], pEasy.Board[i])
	}

	countEasy := 0
	genEasy.solveCountKiller(boardEasy, &countEasy)

	if countEasy != 1 {
		t.Errorf("GenerateKiller('easy') produced a puzzle with %d solutions", countEasy)
	}
}
