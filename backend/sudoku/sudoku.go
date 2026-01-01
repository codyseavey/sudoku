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

// Generate creates a new valid Sudoku puzzle
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

	var g Grid
	// Retry generation until a valid solution is found
	// (Should be first try with standard solver, but kept for robustness)
	for {
		g = make(Grid, gen.N)
		for i := range g {
			g[i] = make([]int, gen.N)
		}

		// Fill the grid using random backtracking
		// We shuffle the numbers 1..N at each step to ensure randomness
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

	// Determine holes
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

	// Determine starting board based on difficulty
	board := make(Grid, gen.N)
	if difficulty == "hard" {
		// Hard Killer Sudoku is typically empty
		for i := range board {
			board[i] = make([]int, gen.N)
		}
	} else {
		// Easy/Medium should have some given digits
		for i := range solution {
			board[i] = make([]int, gen.N)
			copy(board[i], solution[i])
		}

		var holes int
		// Adjust holes logic for Killer Sudoku:
		// Easy Killer -> Standard Medium (50 holes / 31 clues)
		// Medium Killer -> Standard Hard (64 holes / 17 clues)
		if difficulty == "easy" {
			holes = gen.getHolesCount("medium")
		} else {
			holes = gen.getHolesCount("hard")
		}

		gen.removeDigits(board, holes)
	}

	return Puzzle{
		Solution: solution,
		Board:    board,
		Cages:    cages,
		GameType: "killer",
	}
}

func (gen *Generator) generateCages(solution Grid) []Cage {
	visited := make([][]bool, gen.N)
	for i := range visited {
		visited[i] = make([]bool, gen.N)
	}

	var cages []Cage

	for i := 0; i < gen.N; i++ {
		for j := 0; j < gen.N; j++ {
			if !visited[i][j] {
				cage := gen.growCage(solution, visited, i, j)
				cages = append(cages, cage)
			}
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

	// Determine random target size (e.g., 2 to 5)
	targetSize := rand.Intn(4) + 2
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
		default: // medium
			return 20
		}
	}
	// 9x9
	switch difficulty {
	case "easy":
		return 40
	case "hard":
		return 64
	default: // medium
		return 50
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
