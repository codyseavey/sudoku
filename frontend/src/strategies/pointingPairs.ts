import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getBoxCells } from './utils'

/**
 * Pointing Pairs/Triples:
 * If a candidate in a box is restricted to cells that all lie in the same row/column,
 * that candidate can be eliminated from the rest of that row/column outside the box.
 */
export const pointingPairs: Strategy = {
  name: 'Pointing Pairs/Triples',
  difficulty: 1.7,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Iterate through each box
    for (let boxRow = 0; boxRow < size; boxRow += boxHeight) {
      for (let boxCol = 0; boxCol < size; boxCol += boxWidth) {
        const boxCells = getBoxCells(boxRow, boxCol, boxHeight, boxWidth)

        // For each candidate 1 to size
        for (let num = 1; num <= size; num++) {
          // Find all cells in this box that have this candidate
          const cellsWithCandidate: Cell[] = []

          for (const cell of boxCells) {
            if (board[cell.row]?.[cell.col] === null) {
              const cellCands = candidates[cell.row]?.[cell.col] ?? []
              if (cellCands.includes(num)) {
                cellsWithCandidate.push(cell)
              }
            }
          }

          // Need at least 2 cells to form a pointing pair/triple
          if (cellsWithCandidate.length < 2 || cellsWithCandidate.length > 3) {
            continue
          }

          // Check if all cells are in the same row
          const firstCell = cellsWithCandidate[0]!
          const allSameRow = cellsWithCandidate.every(c => c.row === firstCell.row)
          if (allSameRow) {
            const targetRow = firstCell.row
            const eliminations: Elimination[] = []
            const highlights: CellHighlight[] = []

            // Highlight the pointing cells
            for (const cell of cellsWithCandidate) {
              highlights.push({
                row: cell.row,
                col: cell.col,
                type: 'primary',
                candidates: [num]
              })
            }

            // Find eliminations in the rest of the row (outside this box)
            for (let c = 0; c < size; c++) {
              if (c < boxCol || c >= boxCol + boxWidth) {
                if (board[targetRow]?.[c] === null) {
                  const cellCands = candidates[targetRow]?.[c] ?? []
                  if (cellCands.includes(num)) {
                    eliminations.push({ row: targetRow, col: c, candidate: num })
                    highlights.push({
                      row: targetRow,
                      col: c,
                      type: 'elimination',
                      candidates: [num]
                    })
                  }
                }
              }
            }

            if (eliminations.length > 0) {
              const patternName = cellsWithCandidate.length === 2 ? 'Pointing Pair' : 'Pointing Triple'
              const boxNum = Math.floor(boxRow / boxHeight) * (size / boxWidth) + Math.floor(boxCol / boxWidth) + 1

              return {
                strategyName: patternName,
                difficulty: 1.7,
                description: `${num} in box ${boxNum} is restricted to row ${targetRow + 1}. It can be eliminated from other cells in that row.`,
                eliminations,
                highlights
              }
            }
          }

          // Check if all cells are in the same column
          const allSameCol = cellsWithCandidate.every(c => c.col === firstCell.col)
          if (allSameCol) {
            const targetCol = firstCell.col
            const eliminations: Elimination[] = []
            const highlights: CellHighlight[] = []

            // Highlight the pointing cells
            for (const cell of cellsWithCandidate) {
              highlights.push({
                row: cell.row,
                col: cell.col,
                type: 'primary',
                candidates: [num]
              })
            }

            // Find eliminations in the rest of the column (outside this box)
            for (let r = 0; r < size; r++) {
              if (r < boxRow || r >= boxRow + boxHeight) {
                if (board[r]?.[targetCol] === null) {
                  const cellCands = candidates[r]?.[targetCol] ?? []
                  if (cellCands.includes(num)) {
                    eliminations.push({ row: r, col: targetCol, candidate: num })
                    highlights.push({
                      row: r,
                      col: targetCol,
                      type: 'elimination',
                      candidates: [num]
                    })
                  }
                }
              }
            }

            if (eliminations.length > 0) {
              const patternName = cellsWithCandidate.length === 2 ? 'Pointing Pair' : 'Pointing Triple'
              const boxNum = Math.floor(boxRow / boxHeight) * (size / boxWidth) + Math.floor(boxCol / boxWidth) + 1

              return {
                strategyName: patternName,
                difficulty: 1.7,
                description: `${num} in box ${boxNum} is restricted to column ${targetCol + 1}. It can be eliminated from other cells in that column.`,
                eliminations,
                highlights
              }
            }
          }
        }
      }
    }

    return null
  }
}
