package sudoku

// Difficulty levels based on techniques required
const (
	DifficultyEasy    = 1.5 // Only naked/hidden singles
	DifficultyMedium  = 2.5 // May need pointing pairs, subsets
	DifficultyHard    = 3.5 // May need X-Wing, basic fish
	DifficultyExtreme = 4.5 // May need Y-Wing, Skyscraper, etc.
	DifficultyInsane  = 5.0 // Requires multiple advanced techniques
)

// TechniqueResult represents the result of applying a solving technique
type TechniqueResult struct {
	Found      bool
	Difficulty float64
	Name       string
}

// DifficultyAnalysis represents the difficulty analysis of a puzzle
type DifficultyAnalysis struct {
	MaxDifficulty   float64
	TechniquesUsed  []string
	HardMoveCount   int // Number of moves requiring difficulty > 3.5
	Solvable        bool
}

// Candidates represents possible values for each cell
type Candidates [9][9]uint16 // Bitmask: bit i set means i+1 is a candidate

// AnalyzeDifficulty analyzes the difficulty of a puzzle by attempting to solve it
// using various techniques in order of difficulty
func AnalyzeDifficulty(board Grid, size int) DifficultyAnalysis {
	if size != 9 {
		// For 6x6, use simplified analysis
		return analyze6x6Difficulty(board)
	}

	analysis := DifficultyAnalysis{
		TechniquesUsed: []string{},
		Solvable:       false,
	}

	// Create working copy
	work := make(Grid, 9)
	for i := range board {
		work[i] = make([]int, 9)
		copy(work[i], board[i])
	}

	// Initialize candidates
	cands := initCandidates(work)

	// Solve loop - try techniques in order of difficulty
	maxIterations := 500
	for iter := 0; iter < maxIterations; iter++ {
		if isSolved(work) {
			analysis.Solvable = true
			break
		}

		// Try techniques in order of increasing difficulty
		result := tryNakedSingle(work, &cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryHiddenSingle(work, &cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryPointingPairs(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryBoxLineReduction(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryNakedPair(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryHiddenPair(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryNakedTriple(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryXWing(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = trySwordfish(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryYWing(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = trySkyscraper(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		result = tryXYZWing(&cands)
		if result.Found {
			updateAnalysis(&analysis, result)
			continue
		}

		// No technique worked - puzzle requires techniques we don't implement
		// or is unsolvable with logic
		break
	}

	return analysis
}

func analyze6x6Difficulty(board Grid) DifficultyAnalysis {
	// Simplified analysis for 6x6 - just check if solvable with singles
	analysis := DifficultyAnalysis{
		TechniquesUsed: []string{},
		Solvable:       false,
	}

	work := make(Grid, 6)
	for i := range board {
		work[i] = make([]int, 6)
		copy(work[i], board[i])
	}

	// Simple solving with singles only
	for iter := 0; iter < 100; iter++ {
		progress := false
		for r := 0; r < 6; r++ {
			for c := 0; c < 6; c++ {
				if work[r][c] == 0 {
					candidates := getCandidates6x6(work, r, c)
					if len(candidates) == 1 {
						work[r][c] = candidates[0]
						progress = true
						if analysis.MaxDifficulty < 1.0 {
							analysis.MaxDifficulty = 1.0
							analysis.TechniquesUsed = append(analysis.TechniquesUsed, "Naked Single")
						}
					}
				}
			}
		}
		if !progress {
			break
		}
	}

	// Check if solved
	analysis.Solvable = isSolved6x6(work)
	if !analysis.Solvable {
		analysis.MaxDifficulty = 3.0 // Requires advanced techniques
	}

	return analysis
}

func getCandidates6x6(board Grid, row, col int) []int {
	used := make([]bool, 7)
	// Row
	for c := 0; c < 6; c++ {
		if board[row][c] != 0 {
			used[board[row][c]] = true
		}
	}
	// Column
	for r := 0; r < 6; r++ {
		if board[r][col] != 0 {
			used[board[r][col]] = true
		}
	}
	// Box (2x3)
	boxR, boxC := (row/2)*2, (col/3)*3
	for r := boxR; r < boxR+2; r++ {
		for c := boxC; c < boxC+3; c++ {
			if board[r][c] != 0 {
				used[board[r][c]] = true
			}
		}
	}

	var cands []int
	for v := 1; v <= 6; v++ {
		if !used[v] {
			cands = append(cands, v)
		}
	}
	return cands
}

func isSolved6x6(board Grid) bool {
	for r := 0; r < 6; r++ {
		for c := 0; c < 6; c++ {
			if board[r][c] == 0 {
				return false
			}
		}
	}
	return true
}

func updateAnalysis(analysis *DifficultyAnalysis, result TechniqueResult) {
	if result.Difficulty > analysis.MaxDifficulty {
		analysis.MaxDifficulty = result.Difficulty
	}
	if result.Difficulty > 3.5 {
		analysis.HardMoveCount++
	}
	// Track unique techniques
	found := false
	for _, t := range analysis.TechniquesUsed {
		if t == result.Name {
			found = true
			break
		}
	}
	if !found {
		analysis.TechniquesUsed = append(analysis.TechniquesUsed, result.Name)
	}
}

func initCandidates(board Grid) Candidates {
	var cands Candidates
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if board[r][c] == 0 {
				cands[r][c] = 0x1FF // All candidates 1-9
				// Remove used values
				for v := 1; v <= 9; v++ {
					if !canPlace(board, r, c, v) {
						cands[r][c] &^= (1 << (v - 1))
					}
				}
			}
		}
	}
	return cands
}

func canPlace(board Grid, row, col, val int) bool {
	// Check row
	for c := 0; c < 9; c++ {
		if board[row][c] == val {
			return false
		}
	}
	// Check column
	for r := 0; r < 9; r++ {
		if board[r][col] == val {
			return false
		}
	}
	// Check box
	boxR, boxC := (row/3)*3, (col/3)*3
	for r := boxR; r < boxR+3; r++ {
		for c := boxC; c < boxC+3; c++ {
			if board[r][c] == val {
				return false
			}
		}
	}
	return true
}

func isSolved(board Grid) bool {
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if board[r][c] == 0 {
				return false
			}
		}
	}
	return true
}

func countBits(n uint16) int {
	count := 0
	for n > 0 {
		count += int(n & 1)
		n >>= 1
	}
	return count
}

func getOnlyCandidate(mask uint16) int {
	for v := 1; v <= 9; v++ {
		if mask == (1 << (v - 1)) {
			return v
		}
	}
	return 0
}

// placeValue places a value and updates candidates
func placeValue(board Grid, cands *Candidates, row, col, val int) {
	board[row][col] = val
	cands[row][col] = 0
	bit := uint16(1 << (val - 1))

	// Remove from row
	for c := 0; c < 9; c++ {
		cands[row][c] &^= bit
	}
	// Remove from column
	for r := 0; r < 9; r++ {
		cands[r][col] &^= bit
	}
	// Remove from box
	boxR, boxC := (row/3)*3, (col/3)*3
	for r := boxR; r < boxR+3; r++ {
		for c := boxC; c < boxC+3; c++ {
			cands[r][c] &^= bit
		}
	}
}

// tryNakedSingle finds cells with only one candidate
func tryNakedSingle(board Grid, cands *Candidates) TechniqueResult {
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if board[r][c] == 0 && countBits(cands[r][c]) == 1 {
				val := getOnlyCandidate(cands[r][c])
				placeValue(board, cands, r, c, val)
				return TechniqueResult{Found: true, Difficulty: 1.0, Name: "Naked Single"}
			}
		}
	}
	return TechniqueResult{}
}

// tryHiddenSingle finds values that can only go in one cell in a unit
func tryHiddenSingle(board Grid, cands *Candidates) TechniqueResult {
	// Check rows
	for r := 0; r < 9; r++ {
		for v := 1; v <= 9; v++ {
			bit := uint16(1 << (v - 1))
			count := 0
			lastCol := -1
			for c := 0; c < 9; c++ {
				if board[r][c] == 0 && (cands[r][c]&bit) != 0 {
					count++
					lastCol = c
				}
			}
			if count == 1 {
				placeValue(board, cands, r, lastCol, v)
				return TechniqueResult{Found: true, Difficulty: 1.2, Name: "Hidden Single"}
			}
		}
	}

	// Check columns
	for c := 0; c < 9; c++ {
		for v := 1; v <= 9; v++ {
			bit := uint16(1 << (v - 1))
			count := 0
			lastRow := -1
			for r := 0; r < 9; r++ {
				if board[r][c] == 0 && (cands[r][c]&bit) != 0 {
					count++
					lastRow = r
				}
			}
			if count == 1 {
				placeValue(board, cands, lastRow, c, v)
				return TechniqueResult{Found: true, Difficulty: 1.5, Name: "Hidden Single"}
			}
		}
	}

	// Check boxes
	for boxR := 0; boxR < 9; boxR += 3 {
		for boxC := 0; boxC < 9; boxC += 3 {
			for v := 1; v <= 9; v++ {
				bit := uint16(1 << (v - 1))
				count := 0
				lastR, lastC := -1, -1
				for r := boxR; r < boxR+3; r++ {
					for c := boxC; c < boxC+3; c++ {
						if cands[r][c]&bit != 0 {
							count++
							lastR, lastC = r, c
						}
					}
				}
				if count == 1 && lastR >= 0 {
					placeValue(board, cands, lastR, lastC, v)
					return TechniqueResult{Found: true, Difficulty: 1.2, Name: "Hidden Single"}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryPointingPairs finds candidates in a box that are restricted to one row/column
func tryPointingPairs(cands *Candidates) TechniqueResult {
	for boxR := 0; boxR < 9; boxR += 3 {
		for boxC := 0; boxC < 9; boxC += 3 {
			for v := 1; v <= 9; v++ {
				bit := uint16(1 << (v - 1))

				// Find cells in box with this candidate
				var rows, cols []int
				for r := boxR; r < boxR+3; r++ {
					for c := boxC; c < boxC+3; c++ {
						if cands[r][c]&bit != 0 {
							rows = append(rows, r)
							cols = append(cols, c)
						}
					}
				}

				if len(rows) < 2 {
					continue
				}

				// Check if all in same row
				sameRow := true
				for i := 1; i < len(rows); i++ {
					if rows[i] != rows[0] {
						sameRow = false
						break
					}
				}

				if sameRow {
					// Remove from rest of row
					eliminated := false
					for c := 0; c < 9; c++ {
						if c < boxC || c >= boxC+3 {
							if cands[rows[0]][c]&bit != 0 {
								cands[rows[0]][c] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 1.7, Name: "Pointing Pairs"}
					}
				}

				// Check if all in same column
				sameCol := true
				for i := 1; i < len(cols); i++ {
					if cols[i] != cols[0] {
						sameCol = false
						break
					}
				}

				if sameCol {
					// Remove from rest of column
					eliminated := false
					for r := 0; r < 9; r++ {
						if r < boxR || r >= boxR+3 {
							if cands[r][cols[0]]&bit != 0 {
								cands[r][cols[0]] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 1.7, Name: "Pointing Pairs"}
					}
				}
			}
		}
	}
	return TechniqueResult{}
}

// tryBoxLineReduction finds candidates in a row/column restricted to one box
func tryBoxLineReduction(cands *Candidates) TechniqueResult {
	// Check rows
	for r := 0; r < 9; r++ {
		for v := 1; v <= 9; v++ {
			bit := uint16(1 << (v - 1))

			// Find which boxes contain this candidate in this row
			boxMask := 0
			for c := 0; c < 9; c++ {
				if cands[r][c]&bit != 0 {
					boxMask |= (1 << (c / 3))
				}
			}

			// If candidate is only in one box
			if boxMask == 1 || boxMask == 2 || boxMask == 4 {
				boxC := 0
				if boxMask == 2 {
					boxC = 3
				} else if boxMask == 4 {
					boxC = 6
				}

				boxR := (r / 3) * 3
				eliminated := false
				for br := boxR; br < boxR+3; br++ {
					if br != r {
						for bc := boxC; bc < boxC+3; bc++ {
							if cands[br][bc]&bit != 0 {
								cands[br][bc] &^= bit
								eliminated = true
							}
						}
					}
				}
				if eliminated {
					return TechniqueResult{Found: true, Difficulty: 1.7, Name: "Box/Line Reduction"}
				}
			}
		}
	}

	// Check columns
	for c := 0; c < 9; c++ {
		for v := 1; v <= 9; v++ {
			bit := uint16(1 << (v - 1))

			boxMask := 0
			for r := 0; r < 9; r++ {
				if cands[r][c]&bit != 0 {
					boxMask |= (1 << (r / 3))
				}
			}

			if boxMask == 1 || boxMask == 2 || boxMask == 4 {
				boxR := 0
				if boxMask == 2 {
					boxR = 3
				} else if boxMask == 4 {
					boxR = 6
				}

				boxC := (c / 3) * 3
				eliminated := false
				for br := boxR; br < boxR+3; br++ {
					for bc := boxC; bc < boxC+3; bc++ {
						if bc != c && cands[br][bc]&bit != 0 {
							cands[br][bc] &^= bit
							eliminated = true
						}
					}
				}
				if eliminated {
					return TechniqueResult{Found: true, Difficulty: 1.7, Name: "Box/Line Reduction"}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryNakedPair finds two cells in a unit with same two candidates
func tryNakedPair(cands *Candidates) TechniqueResult {
	// Check rows
	for r := 0; r < 9; r++ {
		for c1 := 0; c1 < 9; c1++ {
			if countBits(cands[r][c1]) != 2 {
				continue
			}
			for c2 := c1 + 1; c2 < 9; c2++ {
				if cands[r][c1] == cands[r][c2] {
					// Found naked pair
					eliminated := false
					for c := 0; c < 9; c++ {
						if c != c1 && c != c2 && cands[r][c]&cands[r][c1] != 0 {
							cands[r][c] &^= cands[r][c1]
							eliminated = true
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 2.0, Name: "Naked Pair"}
					}
				}
			}
		}
	}

	// Check columns
	for c := 0; c < 9; c++ {
		for r1 := 0; r1 < 9; r1++ {
			if countBits(cands[r1][c]) != 2 {
				continue
			}
			for r2 := r1 + 1; r2 < 9; r2++ {
				if cands[r1][c] == cands[r2][c] {
					eliminated := false
					for r := 0; r < 9; r++ {
						if r != r1 && r != r2 && cands[r][c]&cands[r1][c] != 0 {
							cands[r][c] &^= cands[r1][c]
							eliminated = true
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 2.0, Name: "Naked Pair"}
					}
				}
			}
		}
	}

	// Check boxes
	for boxR := 0; boxR < 9; boxR += 3 {
		for boxC := 0; boxC < 9; boxC += 3 {
			cells := make([][2]int, 0, 9)
			for r := boxR; r < boxR+3; r++ {
				for c := boxC; c < boxC+3; c++ {
					if countBits(cands[r][c]) == 2 {
						cells = append(cells, [2]int{r, c})
					}
				}
			}
			for i := 0; i < len(cells); i++ {
				for j := i + 1; j < len(cells); j++ {
					r1, c1 := cells[i][0], cells[i][1]
					r2, c2 := cells[j][0], cells[j][1]
					if cands[r1][c1] == cands[r2][c2] {
						eliminated := false
						for r := boxR; r < boxR+3; r++ {
							for c := boxC; c < boxC+3; c++ {
								if (r != r1 || c != c1) && (r != r2 || c != c2) {
									if cands[r][c]&cands[r1][c1] != 0 {
										cands[r][c] &^= cands[r1][c1]
										eliminated = true
									}
								}
							}
						}
						if eliminated {
							return TechniqueResult{Found: true, Difficulty: 2.0, Name: "Naked Pair"}
						}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryHiddenPair finds two candidates that only appear in two cells in a unit
func tryHiddenPair(cands *Candidates) TechniqueResult {
	// Check rows
	for r := 0; r < 9; r++ {
		for v1 := 1; v1 <= 9; v1++ {
			bit1 := uint16(1 << (v1 - 1))
			var cells1 []int
			for c := 0; c < 9; c++ {
				if cands[r][c]&bit1 != 0 {
					cells1 = append(cells1, c)
				}
			}
			if len(cells1) != 2 {
				continue
			}

			for v2 := v1 + 1; v2 <= 9; v2++ {
				bit2 := uint16(1 << (v2 - 1))
				var cells2 []int
				for c := 0; c < 9; c++ {
					if cands[r][c]&bit2 != 0 {
						cells2 = append(cells2, c)
					}
				}
				if len(cells2) == 2 && cells1[0] == cells2[0] && cells1[1] == cells2[1] {
					// Found hidden pair
					pairMask := bit1 | bit2
					eliminated := false
					if cands[r][cells1[0]] != pairMask {
						cands[r][cells1[0]] = pairMask
						eliminated = true
					}
					if cands[r][cells1[1]] != pairMask {
						cands[r][cells1[1]] = pairMask
						eliminated = true
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 2.3, Name: "Hidden Pair"}
					}
				}
			}
		}
	}

	// Check columns
	for c := 0; c < 9; c++ {
		for v1 := 1; v1 <= 9; v1++ {
			bit1 := uint16(1 << (v1 - 1))
			var cells1 []int
			for r := 0; r < 9; r++ {
				if cands[r][c]&bit1 != 0 {
					cells1 = append(cells1, r)
				}
			}
			if len(cells1) != 2 {
				continue
			}

			for v2 := v1 + 1; v2 <= 9; v2++ {
				bit2 := uint16(1 << (v2 - 1))
				var cells2 []int
				for r := 0; r < 9; r++ {
					if cands[r][c]&bit2 != 0 {
						cells2 = append(cells2, r)
					}
				}
				if len(cells2) == 2 && cells1[0] == cells2[0] && cells1[1] == cells2[1] {
					pairMask := bit1 | bit2
					eliminated := false
					if cands[cells1[0]][c] != pairMask {
						cands[cells1[0]][c] = pairMask
						eliminated = true
					}
					if cands[cells1[1]][c] != pairMask {
						cands[cells1[1]][c] = pairMask
						eliminated = true
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 2.3, Name: "Hidden Pair"}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryNakedTriple finds three cells with candidates from same three values
func tryNakedTriple(cands *Candidates) TechniqueResult {
	// Check rows
	for r := 0; r < 9; r++ {
		var cells []int
		for c := 0; c < 9; c++ {
			bits := countBits(cands[r][c])
			if bits >= 2 && bits <= 3 {
				cells = append(cells, c)
			}
		}

		for i := 0; i < len(cells); i++ {
			for j := i + 1; j < len(cells); j++ {
				for k := j + 1; k < len(cells); k++ {
					union := cands[r][cells[i]] | cands[r][cells[j]] | cands[r][cells[k]]
					if countBits(union) == 3 {
						eliminated := false
						for c := 0; c < 9; c++ {
							if c != cells[i] && c != cells[j] && c != cells[k] {
								if cands[r][c]&union != 0 {
									cands[r][c] &^= union
									eliminated = true
								}
							}
						}
						if eliminated {
							return TechniqueResult{Found: true, Difficulty: 2.5, Name: "Naked Triple"}
						}
					}
				}
			}
		}
	}

	// Check columns
	for c := 0; c < 9; c++ {
		var cells []int
		for r := 0; r < 9; r++ {
			bits := countBits(cands[r][c])
			if bits >= 2 && bits <= 3 {
				cells = append(cells, r)
			}
		}

		for i := 0; i < len(cells); i++ {
			for j := i + 1; j < len(cells); j++ {
				for k := j + 1; k < len(cells); k++ {
					union := cands[cells[i]][c] | cands[cells[j]][c] | cands[cells[k]][c]
					if countBits(union) == 3 {
						eliminated := false
						for r := 0; r < 9; r++ {
							if r != cells[i] && r != cells[j] && r != cells[k] {
								if cands[r][c]&union != 0 {
									cands[r][c] &^= union
									eliminated = true
								}
							}
						}
						if eliminated {
							return TechniqueResult{Found: true, Difficulty: 2.5, Name: "Naked Triple"}
						}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryXWing finds X-Wing patterns
func tryXWing(cands *Candidates) TechniqueResult {
	for v := 1; v <= 9; v++ {
		bit := uint16(1 << (v - 1))

		// Check rows for X-Wing
		for r1 := 0; r1 < 9; r1++ {
			var cols1 []int
			for c := 0; c < 9; c++ {
				if cands[r1][c]&bit != 0 {
					cols1 = append(cols1, c)
				}
			}
			if len(cols1) != 2 {
				continue
			}

			for r2 := r1 + 1; r2 < 9; r2++ {
				var cols2 []int
				for c := 0; c < 9; c++ {
					if cands[r2][c]&bit != 0 {
						cols2 = append(cols2, c)
					}
				}
				if len(cols2) == 2 && cols1[0] == cols2[0] && cols1[1] == cols2[1] {
					// Found X-Wing
					eliminated := false
					for r := 0; r < 9; r++ {
						if r != r1 && r != r2 {
							if cands[r][cols1[0]]&bit != 0 {
								cands[r][cols1[0]] &^= bit
								eliminated = true
							}
							if cands[r][cols1[1]]&bit != 0 {
								cands[r][cols1[1]] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 3.2, Name: "X-Wing"}
					}
				}
			}
		}

		// Check columns for X-Wing
		for c1 := 0; c1 < 9; c1++ {
			var rows1 []int
			for r := 0; r < 9; r++ {
				if cands[r][c1]&bit != 0 {
					rows1 = append(rows1, r)
				}
			}
			if len(rows1) != 2 {
				continue
			}

			for c2 := c1 + 1; c2 < 9; c2++ {
				var rows2 []int
				for r := 0; r < 9; r++ {
					if cands[r][c2]&bit != 0 {
						rows2 = append(rows2, r)
					}
				}
				if len(rows2) == 2 && rows1[0] == rows2[0] && rows1[1] == rows2[1] {
					eliminated := false
					for c := 0; c < 9; c++ {
						if c != c1 && c != c2 {
							if cands[rows1[0]][c]&bit != 0 {
								cands[rows1[0]][c] &^= bit
								eliminated = true
							}
							if cands[rows1[1]][c]&bit != 0 {
								cands[rows1[1]][c] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 3.2, Name: "X-Wing"}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// trySwordfish finds Swordfish patterns
func trySwordfish(cands *Candidates) TechniqueResult {
	for v := 1; v <= 9; v++ {
		bit := uint16(1 << (v - 1))

		// Row-based Swordfish
		type rowInfo struct {
			row  int
			cols []int
		}
		var validRows []rowInfo
		for r := 0; r < 9; r++ {
			var cols []int
			for c := 0; c < 9; c++ {
				if cands[r][c]&bit != 0 {
					cols = append(cols, c)
				}
			}
			if len(cols) >= 2 && len(cols) <= 3 {
				validRows = append(validRows, rowInfo{r, cols})
			}
		}

		for i := 0; i < len(validRows); i++ {
			for j := i + 1; j < len(validRows); j++ {
				for k := j + 1; k < len(validRows); k++ {
					colSet := make(map[int]bool)
					for _, c := range validRows[i].cols {
						colSet[c] = true
					}
					for _, c := range validRows[j].cols {
						colSet[c] = true
					}
					for _, c := range validRows[k].cols {
						colSet[c] = true
					}

					if len(colSet) == 3 {
						eliminated := false
						rows := []int{validRows[i].row, validRows[j].row, validRows[k].row}
						rowSet := make(map[int]bool)
						for _, r := range rows {
							rowSet[r] = true
						}

						for c := range colSet {
							for r := 0; r < 9; r++ {
								if !rowSet[r] && cands[r][c]&bit != 0 {
									cands[r][c] &^= bit
									eliminated = true
								}
							}
						}
						if eliminated {
							return TechniqueResult{Found: true, Difficulty: 3.8, Name: "Swordfish"}
						}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryYWing finds Y-Wing patterns
func tryYWing(cands *Candidates) TechniqueResult {
	// Find all bivalue cells
	type biCell struct {
		r, c int
		mask uint16
	}
	var biCells []biCell
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if countBits(cands[r][c]) == 2 {
				biCells = append(biCells, biCell{r, c, cands[r][c]})
			}
		}
	}

	sees := func(r1, c1, r2, c2 int) bool {
		if r1 == r2 {
			return true
		}
		if c1 == c2 {
			return true
		}
		if r1/3 == r2/3 && c1/3 == c2/3 {
			return true
		}
		return false
	}

	for i := 0; i < len(biCells); i++ {
		pivot := biCells[i]
		// pivot has candidates AB

		for j := 0; j < len(biCells); j++ {
			if i == j {
				continue
			}
			wing1 := biCells[j]
			if !sees(pivot.r, pivot.c, wing1.r, wing1.c) {
				continue
			}

			// wing1 must share exactly one candidate with pivot
			shared1 := pivot.mask & wing1.mask
			if countBits(shared1) != 1 {
				continue
			}

			for k := j + 1; k < len(biCells); k++ {
				if i == k {
					continue
				}
				wing2 := biCells[k]
				if !sees(pivot.r, pivot.c, wing2.r, wing2.c) {
					continue
				}
				if sees(wing1.r, wing1.c, wing2.r, wing2.c) {
					continue // Wings shouldn't see each other directly
				}

				shared2 := pivot.mask & wing2.mask
				if countBits(shared2) != 1 {
					continue
				}
				if shared1 == shared2 {
					continue // Wings must share different candidates with pivot
				}

				// The non-shared candidates of wings must be the same
				nonShared1 := wing1.mask &^ pivot.mask
				nonShared2 := wing2.mask &^ pivot.mask
				if nonShared1 != nonShared2 {
					continue
				}

				// Found Y-Wing! Eliminate the common candidate from cells seen by both wings
				eliminated := false
				for r := 0; r < 9; r++ {
					for c := 0; c < 9; c++ {
						if (r == wing1.r && c == wing1.c) || (r == wing2.r && c == wing2.c) {
							continue
						}
						if sees(r, c, wing1.r, wing1.c) && sees(r, c, wing2.r, wing2.c) {
							if cands[r][c]&nonShared1 != 0 {
								cands[r][c] &^= nonShared1
								eliminated = true
							}
						}
					}
				}
				if eliminated {
					return TechniqueResult{Found: true, Difficulty: 4.2, Name: "Y-Wing"}
				}
			}
		}
	}

	return TechniqueResult{}
}

// trySkyscraper finds Skyscraper patterns
func trySkyscraper(cands *Candidates) TechniqueResult {
	for v := 1; v <= 9; v++ {
		bit := uint16(1 << (v - 1))

		// Row-based skyscraper
		type rowPair struct {
			row        int
			col1, col2 int
		}
		var pairs []rowPair
		for r := 0; r < 9; r++ {
			var cols []int
			for c := 0; c < 9; c++ {
				if cands[r][c]&bit != 0 {
					cols = append(cols, c)
				}
			}
			if len(cols) == 2 {
				pairs = append(pairs, rowPair{r, cols[0], cols[1]})
			}
		}

		for i := 0; i < len(pairs); i++ {
			for j := i + 1; j < len(pairs); j++ {
				p1, p2 := pairs[i], pairs[j]
				// One end shares column, other doesn't
				if p1.col1 == p2.col1 && p1.col2 != p2.col2 {
					// Eliminate from cells that see both unshared ends
					eliminated := false
					for r := 0; r < 9; r++ {
						for c := 0; c < 9; c++ {
							if (r == p1.row && c == p1.col2) || (r == p2.row && c == p2.col2) {
								continue
							}
							seesEnd1 := r == p1.row || c == p1.col2 || (r/3 == p1.row/3 && c/3 == p1.col2/3)
							seesEnd2 := r == p2.row || c == p2.col2 || (r/3 == p2.row/3 && c/3 == p2.col2/3)
							if seesEnd1 && seesEnd2 && cands[r][c]&bit != 0 {
								cands[r][c] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 4.0, Name: "Skyscraper"}
					}
				}
				if p1.col2 == p2.col2 && p1.col1 != p2.col1 {
					eliminated := false
					for r := 0; r < 9; r++ {
						for c := 0; c < 9; c++ {
							if (r == p1.row && c == p1.col1) || (r == p2.row && c == p2.col1) {
								continue
							}
							seesEnd1 := r == p1.row || c == p1.col1 || (r/3 == p1.row/3 && c/3 == p1.col1/3)
							seesEnd2 := r == p2.row || c == p2.col1 || (r/3 == p2.row/3 && c/3 == p2.col1/3)
							if seesEnd1 && seesEnd2 && cands[r][c]&bit != 0 {
								cands[r][c] &^= bit
								eliminated = true
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 4.0, Name: "Skyscraper"}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// tryXYZWing finds XYZ-Wing patterns
func tryXYZWing(cands *Candidates) TechniqueResult {
	sees := func(r1, c1, r2, c2 int) bool {
		if r1 == r2 {
			return true
		}
		if c1 == c2 {
			return true
		}
		if r1/3 == r2/3 && c1/3 == c2/3 {
			return true
		}
		return false
	}

	// Find pivot cells with 3 candidates
	for pr := 0; pr < 9; pr++ {
		for pc := 0; pc < 9; pc++ {
			if countBits(cands[pr][pc]) != 3 {
				continue
			}
			pivot := cands[pr][pc]

			// Find wing cells with 2 candidates that are subsets of pivot
			type wingCell struct {
				r, c int
				mask uint16
			}
			var wings []wingCell
			for r := 0; r < 9; r++ {
				for c := 0; c < 9; c++ {
					if r == pr && c == pc {
						continue
					}
					if !sees(pr, pc, r, c) {
						continue
					}
					if countBits(cands[r][c]) == 2 && (cands[r][c]&pivot) == cands[r][c] {
						wings = append(wings, wingCell{r, c, cands[r][c]})
					}
				}
			}

			// Find two wings that together cover all 3 candidates
			for i := 0; i < len(wings); i++ {
				for j := i + 1; j < len(wings); j++ {
					w1, w2 := wings[i], wings[j]
					if w1.mask|w2.mask != pivot {
						continue
					}
					// Common candidate is what both wings share
					common := w1.mask & w2.mask
					if countBits(common) != 1 {
						continue
					}

					// Eliminate common from cells that see pivot and both wings
					eliminated := false
					for r := 0; r < 9; r++ {
						for c := 0; c < 9; c++ {
							if (r == pr && c == pc) || (r == w1.r && c == w1.c) || (r == w2.r && c == w2.c) {
								continue
							}
							if sees(r, c, pr, pc) && sees(r, c, w1.r, w1.c) && sees(r, c, w2.r, w2.c) {
								if cands[r][c]&common != 0 {
									cands[r][c] &^= common
									eliminated = true
								}
							}
						}
					}
					if eliminated {
						return TechniqueResult{Found: true, Difficulty: 4.4, Name: "XYZ-Wing"}
					}
				}
			}
		}
	}

	return TechniqueResult{}
}

// GetDifficultyLevel returns the difficulty level name based on analysis
func GetDifficultyLevel(analysis DifficultyAnalysis) string {
	if !analysis.Solvable {
		return "unknown"
	}
	if analysis.MaxDifficulty <= DifficultyEasy {
		return "easy"
	}
	if analysis.MaxDifficulty <= DifficultyMedium {
		return "medium"
	}
	if analysis.MaxDifficulty <= DifficultyHard {
		return "hard"
	}
	if analysis.MaxDifficulty <= DifficultyExtreme {
		return "extreme"
	}
	return "insane"
}

// MatchesDifficulty checks if a puzzle matches the requested difficulty
func MatchesDifficulty(analysis DifficultyAnalysis, targetDifficulty string) bool {
	if !analysis.Solvable {
		return false
	}

	switch targetDifficulty {
	case "easy":
		// Must be solvable with only singles (difficulty <= 1.5)
		return analysis.MaxDifficulty <= DifficultyEasy
	case "medium":
		// Requires some intermediate techniques (1.5 < difficulty <= 2.5)
		return analysis.MaxDifficulty > DifficultyEasy && analysis.MaxDifficulty <= DifficultyMedium
	case "hard":
		// Requires at least one moderately difficult technique (2.5 < difficulty <= 3.5)
		return analysis.MaxDifficulty > DifficultyMedium && analysis.MaxDifficulty <= DifficultyHard
	case "extreme":
		// Requires advanced techniques (3.5 < difficulty <= 4.5)
		return analysis.MaxDifficulty > DifficultyHard && analysis.MaxDifficulty <= DifficultyExtreme
	case "insane":
		// Requires multiple of the hardest techniques (difficulty > 4.5 OR multiple hard moves)
		return analysis.MaxDifficulty > DifficultyExtreme || analysis.HardMoveCount >= 3
	default:
		return true
	}
}
