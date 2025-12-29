package sudoku

import (
	"math/rand"
	"time"
)

const N = 9
const SRN = 3 // Square Root of N

type Grid [N][N]int

type Puzzle struct {
	Solution Grid `json:"solution"`
	Board    Grid `json:"board"`
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Generate creates a new valid Sudoku puzzle with a unique solution
func Generate(difficulty string) Puzzle {
	var g Grid
	g.fillDiagonal()
	g.fillRemaining(0, SRN)

	solution := g // Store the full solution

	// Determine number of holes based on difficulty
	var k int
	switch difficulty {
	case "easy":
		k = 30
	case "hard":
		k = 50
	case "medium":
		fallthrough
	default:
		k = 40
	}

	puzzleBoard := g
	puzzleBoard.removeDigits(k)

	return Puzzle{
		Solution: solution,
		Board:    puzzleBoard,
	}
}

func (g *Grid) fillDiagonal() {
	for i := 0; i < N; i = i + SRN {
		g.fillBox(i, i)
	}
}

func (g *Grid) fillBox(row, col int) {
	num := 0
	for i := 0; i < SRN; i++ {
		for j := 0; j < SRN; j++ {
			for {
				num = rand.Intn(N) + 1
				if g.isSafeBox(row, col, num) {
					break
				}
			}
			g[row+i][col+j] = num
		}
	}
}

func (g *Grid) isSafeBox(rowStart, colStart, num int) bool {
	for i := 0; i < SRN; i++ {
		for j := 0; j < SRN; j++ {
			if g[rowStart+i][colStart+j] == num {
				return false
			}
		}
	}
	return true
}

func (g *Grid) isSafe(row, col, num int) bool {
	return g.unUsedInRow(row, num) &&
		g.unUsedInCol(col, num) &&
		g.unUsedInBox(row-row%SRN, col-col%SRN, num)
}

func (g *Grid) unUsedInRow(row, num int) bool {
	for j := 0; j < N; j++ {
		if g[row][j] == num {
			return false
		}
	}
	return true
}

func (g *Grid) unUsedInCol(col, num int) bool {
	for i := 0; i < N; i++ {
		if g[i][col] == num {
			return false
		}
	}
	return true
}

func (g *Grid) unUsedInBox(rowStart, colStart, num int) bool {
	for i := 0; i < SRN; i++ {
		for j := 0; j < SRN; j++ {
			if g[rowStart+i][colStart+j] == num {
				return false
			}
		}
	}
	return true
}

func (g *Grid) fillRemaining(i, j int) bool {
	if j >= N && i < N-1 {
		i = i + 1
		j = 0
	}
	if i >= N && j >= N {
		return true
	}
	if i < SRN {
		if j < SRN {
			j = SRN
		}
	} else if i < N-SRN {
		if j == (i/SRN)*SRN {
			j = j + SRN
		}
	} else {
		if j == N-SRN {
			i = i + 1
			j = 0
			if i >= N {
				return true
			}
		}
	}

	for num := 1; num <= N; num++ {
		if g.isSafe(i, j, num) {
			g[i][j] = num
			if g.fillRemaining(i, j+1) {
				return true
			}
			g[i][j] = 0
		}
	}
	return false
}

// removeDigits tries to remove K digits while maintaining a unique solution.
func (g *Grid) removeDigits(k int) {
	count := k
	for count != 0 {
		cellId := rand.Intn(N * N)
		i := cellId / N
		j := cellId % N
		if g[i][j] != 0 {
			backup := g[i][j]
			g[i][j] = 0

			// Check if solution is unique
			solutions := 0
			g.solveCount(&solutions)
			
			if solutions != 1 {
				g[i][j] = backup // Put it back if not unique
			} else {
				count--
			}
		}
	}
}

func (g *Grid) solveCount(count *int) {
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			if g[i][j] == 0 {
				for num := 1; num <= N; num++ {
					if g.isSafe(i, j, num) {
						g[i][j] = num
						g.solveCount(count)
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
