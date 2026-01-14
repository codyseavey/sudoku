package sudoku

import (
	"testing"
)

func TestGenerateKillerDifficulty(t *testing.T) {
	size := 9

	// Test Easy
	pEasy := GenerateKiller("easy", size)
	filledCountEasy := 0
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if pEasy.Board[r][c] != 0 {
				filledCountEasy++
			}
		}
	}
	// Expect some digits (approx 31 clues)
	if filledCountEasy == 0 {
		t.Error("Killer Sudoku Easy should have clues")
	}

	// Test Medium
	pMed := GenerateKiller("medium", size)
	filledCountMed := 0
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if pMed.Board[r][c] != 0 {
				filledCountMed++
			}
		}
	}
	// Expect some digits (approx 17 clues)
	if filledCountMed == 0 {
		t.Error("Killer Sudoku Medium should have clues")
	}

	// Test Hard
	pHard := GenerateKiller("hard", size)
	filledCountHard := 0
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if pHard.Board[r][c] != 0 {
				filledCountHard++
			}
		}
	}
	// Hard should have fewer clues than medium
	if filledCountHard == 0 {
		t.Error("Killer Sudoku Hard should have clues")
	}

	// Test Insane (should have very few clues)
	pInsane := GenerateKiller("insane", size)
	filledCountInsane := 0
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if pInsane.Board[r][c] != 0 {
				filledCountInsane++
			}
		}
	}
	// Expect very few digits (insane puzzles may have some clues depending on cage structure)
	// Based on getHolesRange, insane allows 58-66 holes, meaning 15-23 clues
	if filledCountInsane > 20 {
		t.Errorf("Killer Sudoku Insane should have very few clues, got %d clues", filledCountInsane)
	}
}

func TestGenerateKillerStructure(t *testing.T) {
	p := GenerateKiller("easy", 9)

	if p.GameType != "killer" {
		t.Errorf("Expected GameType 'killer', got %s", p.GameType)
	}
	if len(p.Cages) == 0 {
		t.Error("Expected Cages to be generated")
	}

	// Verify all cells covered by exactly one cage
	covered := make([][]int, 9)
	for i := range covered {
		covered[i] = make([]int, 9)
	}

	for idx, cage := range p.Cages {
		expectedSum := 0
		for _, cell := range cage.Cells {
			covered[cell.Row][cell.Col]++
			expectedSum += p.Solution[cell.Row][cell.Col]
		}
		if cage.Sum != expectedSum {
			t.Errorf("Cage %d sum mismatch: expected %d, got %d", idx, expectedSum, cage.Sum)
		}
	}

	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if covered[r][c] != 1 {
				t.Errorf("Cell (%d, %d) covered %d times", r, c, covered[r][c])
			}
		}
	}
}
