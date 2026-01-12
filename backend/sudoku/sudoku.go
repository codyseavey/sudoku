package sudoku

import (
	"math/rand"
	"time"
)

type Grid [][]int

type Point struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

type Cage struct {
	Sum   int     `json:"sum"`
	Cells []Point `json:"cells"`
}

type Puzzle struct {
	Solution Grid   `json:"solution"`
	Board    Grid   `json:"board"`
	Cages    []Cage `json:"cages,omitempty"`
	GameType string `json:"gameType,omitempty"`
}

type Generator struct {
	N         int
	BoxHeight int
	BoxWidth  int
	Cages     []Cage
	cageMap   map[Point]int
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Solve attempts to solve a given partial grid
func Solve(initialBoard Grid, size int) (Grid, bool) {
	gen := Generator{N: size}
	
	if size == 6 {
		gen.BoxHeight = 2
		gen.BoxWidth = 3
	} else {
		gen.N = 9
		gen.BoxHeight = 3
		gen.BoxWidth = 3
	}

	// 1. Validate initial board state
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			num := initialBoard[i][j]
			if num != 0 {
				// Temporarily clear cell to check if it's safe in its position
				initialBoard[i][j] = 0
				if !gen.isSafe(initialBoard, i, j, num) {
					return nil, false // Invalid initial state
				}
				initialBoard[i][j] = num
			}
		}
	}

	// 2. Deep copy to work on
	solution := make(Grid, gen.N)
	for i := range initialBoard {
		solution[i] = make([]int, gen.N)
		copy(solution[i], initialBoard[i])
	}

	// 3. Check for uniqueness
	// We need to count solutions. If > 1, it's invalid for a custom puzzle.
	count := 0
	
	// solveCount modifies the grid, so use a copy for counting if we want to preserve 'solution' for the final return
	// Actually, solveCount will zero out the grid as it backtracks.
	// So we should run it on 'solution' copy.
	
	countGrid := make(Grid, gen.N)
	for i := range initialBoard {
		countGrid[i] = make([]int, gen.N)
		copy(countGrid[i], initialBoard[i])
	}
	
	gen.solveCount(countGrid, &count)
	
	if count == 0 {
		return nil, false // No solution
	}
	if count > 1 {
		return nil, false // Multiple solutions
	}

	// 4. Fill the single solution for return
	// Since count == 1, we can just run fillGrid to get it.
	if gen.fillGrid(solution) {
		return solution, true
	}

	return nil, false
}

// Generate creates a new valid Sudoku puzzle with technique-based difficulty
func Generate(difficulty string, size int) Puzzle {
	gen := Generator{N: size}

	// Configure dimensions
	if size == 6 {
		gen.BoxHeight = 2
		gen.BoxWidth = 3
	} else {
		// Default to 9x9
		gen.N = 9
		gen.BoxHeight = 3
		gen.BoxWidth = 3
	}

	minHoles, maxHoles := gen.getHolesRange(difficulty)
	maxAttempts := 50 // Limit attempts to avoid infinite loops

	for attempt := 0; attempt < maxAttempts; attempt++ {
		var g Grid
		// Generate a complete valid grid
		for {
			g = make(Grid, gen.N)
			for i := range g {
				g[i] = make([]int, gen.N)
			}
			if gen.fillGrid(g) {
				break
			}
		}

		// Deep copy for solution
		solution := make(Grid, gen.N)
		for i := range g {
			solution[i] = make([]int, gen.N)
			copy(solution[i], g[i])
		}

		// Try different hole counts within the range
		holes := minHoles + rand.Intn(maxHoles-minHoles+1)

		puzzleBoard := make(Grid, gen.N)
		for i := range g {
			puzzleBoard[i] = make([]int, gen.N)
			copy(puzzleBoard[i], g[i])
		}

		gen.removeDigits(puzzleBoard, holes)

		// Analyze difficulty
		analysis := AnalyzeDifficulty(puzzleBoard, gen.N)

		// Check if puzzle matches target difficulty
		if MatchesDifficulty(analysis, difficulty) {
			return Puzzle{
				Solution: solution,
				Board:    puzzleBoard,
			}
		}
	}

	// Fallback: return a puzzle with default hole count if we couldn't find a matching one
	var g Grid
	for {
		g = make(Grid, gen.N)
		for i := range g {
			g[i] = make([]int, gen.N)
		}
		if gen.fillGrid(g) {
			break
		}
	}

	solution := make(Grid, gen.N)
	for i := range g {
		solution[i] = make([]int, gen.N)
		copy(solution[i], g[i])
	}

	k := gen.getHolesCount(difficulty)
	puzzleBoard := make(Grid, gen.N)
	for i := range g {
		puzzleBoard[i] = make([]int, gen.N)
		copy(puzzleBoard[i], g[i])
	}
	gen.removeDigits(puzzleBoard, k)

	return Puzzle{
		Solution: solution,
		Board:    puzzleBoard,
	}
}

// GenerateKiller creates a new valid Killer Sudoku puzzle
func GenerateKiller(difficulty string, size int) Puzzle {
	gen := Generator{N: size}

	// Configure dimensions
	if size == 6 {
		gen.BoxHeight = 2
		gen.BoxWidth = 3
	} else {
		// Default to 9x9
		gen.N = 9
		gen.BoxHeight = 3
		gen.BoxWidth = 3
	}

	var solution Grid
	// Retry generation until a valid solution is found
	for {
		solution = make(Grid, gen.N)
		for i := range solution {
			solution[i] = make([]int, gen.N)
		}
		if gen.fillGrid(solution) {
			break
		}
	}

	// Generate Cages
	cages := gen.generateCages(solution)
	gen.Cages = cages // Set cages for solver

	// Build fast lookup map for cages
	gen.cageMap = make(map[Point]int)
	for idx, cage := range cages {
		for _, cell := range cage.Cells {
			gen.cageMap[cell] = idx
		}
	}

	board := make(Grid, gen.N)
	for i := range solution {
		board[i] = make([]int, gen.N)
		copy(board[i], solution[i])
	}

	var holes int
	switch difficulty {
	case "insane":
		holes = gen.N * gen.N // Try to remove all
	case "extreme":
		holes = gen.N*gen.N - 5 // Remove almost all
	case "hard":
		holes = gen.getHolesCount("hard")
	case "easy":
		holes = gen.getHolesCount("medium")
	default: // medium
		holes = gen.getHolesCount("hard")
	}
	
	gen.removeDigitsKiller(board, holes)

	return Puzzle{
		Solution: solution,
		Board:    board,
		Cages:    cages,
		GameType: "killer",
	}
}

func (gen *Generator) generateCages(solution Grid) []Cage {
	// Retry cage generation a few times if we get stuck (rare but possible)
	for attempt := 0; attempt < 10; attempt++ {
		visited := make([][]bool, gen.N)
		for i := range visited {
			visited[i] = make([]bool, gen.N)
		}

		var cages []Cage
		success := true

		for i := 0; i < gen.N; i++ {
			for j := 0; j < gen.N; j++ {
				if !visited[i][j] {
					cage := gen.growCage(solution, visited, i, j)
					if len(cage.Cells) == 0 {
						success = false
						break
					}
					cages = append(cages, cage)
				}
			}
			if !success {
				break
			}
		}
		if success {
			return cages
		}
	}
	// Fallback: simple 1-cell cages if generation fails (should happen very rarely)
	var cages []Cage
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			cages = append(cages, Cage{Sum: solution[i][j], Cells: []Point{{i, j}}})
		}
	}
	return cages
}

func (gen *Generator) growCage(solution Grid, visited [][]bool, r, c int) Cage {
	// Initialize cage with starting cell
	cells := []Point{{r, c}}
	visited[r][c] = true
	values := make(map[int]bool)
	values[solution[r][c]] = true

	// Determine random target size (e.g., 2 to 9 for training data)
	targetSize := rand.Intn(8) + 2
	if gen.N == 6 {
		targetSize = rand.Intn(3) + 2
	}

	// Directions: up, down, left, right
	dirs := []Point{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}

	for len(cells) < targetSize {
		// Pick a random cell from current cage to expand from
		perm := rand.Perm(len(cells))
		expanded := false

		for _, idx := range perm {
			curr := cells[idx]

			// Shuffle directions
			dirPerm := rand.Perm(4)
			for _, dIdx := range dirPerm {
				d := dirs[dIdx]
				nr, nc := curr.Row+d.Row, curr.Col+d.Col

				// Check bounds
				if nr >= 0 && nr < gen.N && nc >= 0 && nc < gen.N {
					if !visited[nr][nc] {
						val := solution[nr][nc]
						if !values[val] {
							// Add to cage
							visited[nr][nc] = true
							cells = append(cells, Point{nr, nc})
							values[val] = true
							expanded = true
							break
						}
					}
				}
			}
			if expanded {
				break
			}
		}

		if !expanded {
			break
		}
	}

	// Calculate Sum
	sum := 0
	for _, cell := range cells {
		sum += solution[cell.Row][cell.Col]
	}

	return Cage{Sum: sum, Cells: cells}
}

func (gen *Generator) getHolesCount(difficulty string) int {
	if gen.N == 6 {
		switch difficulty {
		case "easy":
			return 15
		case "hard":
			return 25
		case "extreme":
			return 28
		case "insane":
			return 30
		default: // medium
			return 20
		}
	}
	// 9x9
	switch difficulty {
	case "easy":
		return 40
	case "hard":
		return 55
	case "extreme":
		return 60
	case "insane":
		return 64
	default: // medium
		return 48
	}
}

// getHolesRange returns min and max holes to try for a difficulty level
func (gen *Generator) getHolesRange(difficulty string) (int, int) {
	if gen.N == 6 {
		switch difficulty {
		case "easy":
			return 12, 18
		case "medium":
			return 16, 22
		case "hard":
			return 20, 26
		case "extreme":
			return 24, 28
		case "insane":
			return 26, 32
		default:
			return 16, 22
		}
	}
	// 9x9
	switch difficulty {
	case "easy":
		return 35, 45
	case "medium":
		return 42, 52
	case "hard":
		return 48, 58
	case "extreme":
		return 54, 62
	case "insane":
		return 58, 66
	default:
		return 42, 52
	}
}

// fillGrid fills the entire grid using backtracking with randomized candidates
func (gen *Generator) fillGrid(g Grid) bool {
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			if g[i][j] == 0 {
				nums := rand.Perm(gen.N)
				for _, n := range nums {
					num := n + 1 // Perm returns 0..N-1
					if gen.isSafe(g, i, j, num) {
						g[i][j] = num
						if gen.fillGrid(g) {
							return true
						}
						g[i][j] = 0
					}
				}
				return false
			}
		}
	}
	return true
}

func (gen *Generator) isSafe(g Grid, row, col, num int) bool {
	return gen.unUsedInRow(g, row, num) &&
		gen.unUsedInCol(g, col, num) &&
		gen.unUsedInBox(g, row-row%gen.BoxHeight, col-col%gen.BoxWidth, num)
}

func (gen *Generator) unUsedInRow(g Grid, row, num int) bool {
	for j := 0; j < gen.N; j++ {
		if g[row][j] == num {
			return false
		}
	}
	return true
}

func (gen *Generator) unUsedInCol(g Grid, col, num int) bool {
	for i := 0; i < gen.N; i++ {
		if g[i][col] == num {
			return false
		}
	}
	return true
}

func (gen *Generator) unUsedInBox(g Grid, rowStart, colStart, num int) bool {
	for i := 0; i < gen.BoxHeight; i++ {
		for j := 0; j < gen.BoxWidth; j++ {
			if g[rowStart+i][colStart+j] == num {
				return false
			}
		}
	}
	return true
}

func (gen *Generator) removeDigits(g Grid, k int) {
	type point struct{ r, c int }
	cells := make([]point, 0, gen.N*gen.N)
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			cells = append(cells, point{i, j})
		}
	}

	rand.Shuffle(len(cells), func(i, j int) {
		cells[i], cells[j] = cells[j], cells[i]
	})

	count := k
	for _, cell := range cells {
		if count <= 0 {
			break
		}

		i, j := cell.r, cell.c
		if g[i][j] != 0 {
			backup := g[i][j]
			g[i][j] = 0

			solutions := 0
			gen.solveCount(g, &solutions)

			if solutions != 1 {
				g[i][j] = backup
			} else {
				count--
			}
		}
	}
}

func (gen *Generator) removeDigitsKiller(g Grid, k int) {
	type point struct{ r, c int }
	cells := make([]point, 0, gen.N*gen.N)
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			cells = append(cells, point{i, j})
		}
	}

	rand.Shuffle(len(cells), func(i, j int) {
		cells[i], cells[j] = cells[j], cells[i]
	})

	count := k
	for _, cell := range cells {
		if count <= 0 {
			break
		}

		i, j := cell.r, cell.c
		if g[i][j] != 0 {
			backup := g[i][j]
			g[i][j] = 0

			solutions := 0
			steps := 0
			// Limit steps to ~200,000 to prevent hangs. 
			// If it takes longer, we assume we can't verify uniqueness cheaply, so we keep the clue.
			maxSteps := 200000 
			gen.solveCountKiller(g, &solutions, &steps, maxSteps)

			if solutions != 1 {
				g[i][j] = backup
			} else {
				count--
			}
		}
	}
}

func (gen *Generator) solveCount(g Grid, count *int) {
	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			if g[i][j] == 0 {
				for num := 1; num <= gen.N; num++ {
					if gen.isSafe(g, i, j, num) {
						g[i][j] = num
						gen.solveCount(g, count)
						g[i][j] = 0
						if *count > 1 {
							return
						}
					}
				}
				return
			}
		}
	}
	*count++
}

func (gen *Generator) solveCountKiller(g Grid, count *int, steps *int, maxSteps int) {
	*steps++
	if *steps > maxSteps {
		*count = 2 // Force stop, treat as non-unique (conservative)
		return
	}

	type Move struct {
		r, c       int
		candidates []int
	}

	// Find the cell with the fewest candidates (MRV)
	var bestMove *Move
	minCandidates := gen.N + 1

	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			if g[i][j] == 0 {
				var cands []int
				for num := 1; num <= gen.N; num++ {
					if gen.isSafeKiller(g, i, j, num) {
						cands = append(cands, num)
					}
				}

				if len(cands) == 0 {
					return // Dead end
				}

				if len(cands) < minCandidates {
					minCandidates = len(cands)
					bestMove = &Move{r: i, c: j, candidates: cands}
				}
			}
		}
	}

	// No empty cells found, solution found
	if bestMove == nil {
		*count++
		return
	}

	// Try candidates for the best cell
	r, c := bestMove.r, bestMove.c
	for _, num := range bestMove.candidates {
		g[r][c] = num
		gen.solveCountKiller(g, count, steps, maxSteps)
		g[r][c] = 0
		if *count > 1 {
			return
		}
	}
}

func (gen *Generator) isSafeKiller(g Grid, row, col, num int) bool {
	// 1. Standard Sudoku Checks
	if !gen.isSafe(g, row, col, num) {
		return false
	}

	// 2. Killer Sudoku Checks (Cages)
	// Find the cage this cell belongs to
	var currentCage *Cage
	if idx, ok := gen.cageMap[Point{row, col}]; ok {
		currentCage = &gen.Cages[idx]
	}

	if currentCage == nil {
		return true // Should not happen if cages cover all cells
	}

	currentSum := 0
	filledCount := 0
	used := 0 // Bitmask for used numbers in cage

	for _, cell := range currentCage.Cells {
		val := g[cell.Row][cell.Col]
		// If it's the cell we are currently placing (row, col), use 'num'
		if cell.Row == row && cell.Col == col {
			val = num
		}

		if val != 0 {
			// Check for duplicates in cage
			if val == num && (cell.Row != row || cell.Col != col) {
				// We found 'num' elsewhere in the cage
				return false
			}
			currentSum += val
			filledCount++
			used |= (1 << val)
		}
	}

	// Check if sum exceeded
	if currentSum > currentCage.Sum {
		return false
	}

	// If cage is full, sum must match exactly
	remainingCells := len(currentCage.Cells) - filledCount
	remainingSum := currentCage.Sum - currentSum

	if remainingCells == 0 {
		if remainingSum != 0 {
			return false
		}
	} else {
		// Optimization: Check if remaining sum is achievable with remaining distinct numbers
		
		// 1. Minimum possible sum check
		minSum := 0
		count := 0
		for v := 1; v <= gen.N; v++ {
			if (used & (1 << v)) == 0 {
				minSum += v
				count++
				if count == remainingCells {
					break
				}
			}
		}
		if remainingSum < minSum {
			return false
		}

		// 2. Maximum possible sum check
		maxSum := 0
		count = 0
		for v := gen.N; v >= 1; v-- {
			if (used & (1 << v)) == 0 {
				maxSum += v
				count++
				if count == remainingCells {
					break
				}
			}
		}
		if remainingSum > maxSum {
			return false
		}
	}

	return true
}
