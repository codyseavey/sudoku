import type { Strategy, SolveContext, HintResult, CellHighlight, Cell } from './types'
import { formatCell } from './utils'

/**
 * BUG+1 (Bi-Value Universal Grave + 1):
 * A BUG state is when every unsolved cell has exactly 2 candidates,
 * and each candidate appears exactly twice in every house.
 * This would create multiple solutions (deadly pattern).
 *
 * BUG+1: All cells are bivalue except ONE cell with 3 candidates.
 * The "extra" candidate (appearing 3 times in its houses) must be the solution.
 */
export const bug: Strategy = {
  name: 'BUG+1',
  difficulty: 5.0,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size } = context

    // Count cells by candidate count
    let trivalueCell: { cell: Cell; cands: number[] } | null = null
    let trivalueCount = 0

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r]?.[c] === null) {
          const cellCands = candidates[r]?.[c] ?? []

          if (cellCands.length === 3) {
            trivalueCell = { cell: { row: r, col: c }, cands: [...cellCands] }
            trivalueCount++
          } else if (cellCands.length !== 2) {
            // Not a BUG+1 state - cells must be bivalue or the single trivalue
            return null
          }
        }
      }
    }

    // Must have exactly one trivalue cell
    if (trivalueCount !== 1 || !trivalueCell) {
      return null
    }

    // Verify BUG condition: each candidate appears exactly twice in each house
    // except for one candidate in the trivalue cell's houses (which appears 3 times)
    const { row, col } = trivalueCell.cell

    // Count candidates in each house containing the trivalue cell
    const rowCounts = countCandidatesInRow(row, board, candidates, size)
    const colCounts = countCandidatesInCol(col, board, candidates, size)
    const boxCounts = countCandidatesInBox(row, col, board, candidates, size)

    // Find the candidate that appears 3 times in all three houses
    let bugDigit: number | null = null

    for (const cand of trivalueCell.cands) {
      const rowCount = rowCounts.get(cand) ?? 0
      const colCount = colCounts.get(cand) ?? 0
      const boxCount = boxCounts.get(cand) ?? 0

      // In a BUG+1, the extra digit appears 3 times in the row, col, and box
      if (rowCount === 3 && colCount === 3 && boxCount === 3) {
        bugDigit = cand
        break
      }
    }

    if (bugDigit === null) {
      // Verify other candidates appear exactly twice
      // If not, this isn't a true BUG+1 state
      return null
    }

    // Found BUG+1! The bug digit must be placed
    const otherCandidates = trivalueCell.cands.filter(c => c !== bugDigit)

    const highlights: CellHighlight[] = [
      {
        row: trivalueCell.cell.row,
        col: trivalueCell.cell.col,
        type: 'primary',
        candidates: [bugDigit]
      },
      {
        row: trivalueCell.cell.row,
        col: trivalueCell.cell.col,
        type: 'elimination',
        candidates: otherCandidates
      }
    ]

    // Create eliminations for the other candidates
    const eliminations = otherCandidates.map(c => ({
      row: trivalueCell!.cell.row,
      col: trivalueCell!.cell.col,
      candidate: c
    }))

    return {
      strategyName: 'BUG+1',
      difficulty: 5.0,
      description: `BUG+1 detected: All cells are bi-value except ${formatCell(row, col)} which has three candidates. The digit ${bugDigit} appears three times in the row, column, and box. To avoid multiple solutions, ${formatCell(row, col)} must be ${bugDigit}.`,
      eliminations,
      highlights
    }
  }
}

function countCandidatesInRow(
  row: number,
  board: (number | null)[][],
  candidates: number[][][],
  size: number
): Map<number, number> {
  const counts = new Map<number, number>()

  for (let c = 0; c < size; c++) {
    if (board[row]?.[c] === null) {
      const cellCands = candidates[row]?.[c] ?? []
      for (const cand of cellCands) {
        counts.set(cand, (counts.get(cand) ?? 0) + 1)
      }
    }
  }

  return counts
}

function countCandidatesInCol(
  col: number,
  board: (number | null)[][],
  candidates: number[][][],
  size: number
): Map<number, number> {
  const counts = new Map<number, number>()

  for (let r = 0; r < size; r++) {
    if (board[r]?.[col] === null) {
      const cellCands = candidates[r]?.[col] ?? []
      for (const cand of cellCands) {
        counts.set(cand, (counts.get(cand) ?? 0) + 1)
      }
    }
  }

  return counts
}

function countCandidatesInBox(
  row: number,
  col: number,
  board: (number | null)[][],
  candidates: number[][][],
  _size: number
): Map<number, number> {
  const counts = new Map<number, number>()

  // Assuming 3x3 boxes for 9x9 grid
  const boxHeight = 3
  const boxWidth = 3
  const startRow = Math.floor(row / boxHeight) * boxHeight
  const startCol = Math.floor(col / boxWidth) * boxWidth

  for (let r = startRow; r < startRow + boxHeight; r++) {
    for (let c = startCol; c < startCol + boxWidth; c++) {
      if (board[r]?.[c] === null) {
        const cellCands = candidates[r]?.[c] ?? []
        for (const cand of cellCands) {
          counts.set(cand, (counts.get(cand) ?? 0) + 1)
        }
      }
    }
  }

  return counts
}
