import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getRowCells, getColCells, getBoxStart, getBoxCells } from './utils'

/**
 * Box/Line Reduction (Claiming):
 * If a candidate in a row/column is restricted to cells that all lie within one box,
 * that candidate can be eliminated from the rest of that box.
 * This is the inverse of Pointing Pairs.
 */
export const boxLineReduction: Strategy = {
  name: 'Box/Line Reduction',
  difficulty: 1.7,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Check rows
    for (let row = 0; row < size; row++) {
      const result = checkLine('row', row, board, candidates, size, boxHeight, boxWidth)
      if (result) return result
    }

    // Check columns
    for (let col = 0; col < size; col++) {
      const result = checkLine('col', col, board, candidates, size, boxHeight, boxWidth)
      if (result) return result
    }

    return null
  }
}

function checkLine(
  lineType: 'row' | 'col',
  lineIndex: number,
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  const lineCells = lineType === 'row'
    ? getRowCells(lineIndex, size)
    : getColCells(lineIndex, size)

  // For each candidate 1 to size
  for (let num = 1; num <= size; num++) {
    // Find all cells in this line that have this candidate
    const cellsWithCandidate: Cell[] = []

    for (const cell of lineCells) {
      if (board[cell.row]?.[cell.col] === null) {
        const cellCands = candidates[cell.row]?.[cell.col] ?? []
        if (cellCands.includes(num)) {
          cellsWithCandidate.push(cell)
        }
      }
    }

    // Need at least 2 cells
    if (cellsWithCandidate.length < 2) continue

    // Check if all cells are in the same box
    const firstCell = cellsWithCandidate[0]!
    const firstBox = getBoxStart(
      firstCell.row,
      firstCell.col,
      boxHeight,
      boxWidth
    )

    const allInSameBox = cellsWithCandidate.every(cell => {
      const box = getBoxStart(cell.row, cell.col, boxHeight, boxWidth)
      return box.boxStartRow === firstBox.boxStartRow && box.boxStartCol === firstBox.boxStartCol
    })

    if (!allInSameBox) continue

    // All cells with this candidate are in the same box
    // Find eliminations in the rest of the box
    const boxCells = getBoxCells(firstBox.boxStartRow, firstBox.boxStartCol, boxHeight, boxWidth)
    const eliminations: Elimination[] = []
    const highlights: CellHighlight[] = []

    // Highlight the claiming cells
    for (const cell of cellsWithCandidate) {
      highlights.push({
        row: cell.row,
        col: cell.col,
        type: 'primary',
        candidates: [num]
      })
    }

    // Find cells in the box not on this line that have this candidate
    for (const cell of boxCells) {
      // Skip cells on the line
      const isOnLine = lineType === 'row'
        ? cell.row === lineIndex
        : cell.col === lineIndex

      if (isOnLine) continue

      if (board[cell.row]?.[cell.col] === null) {
        const cellCands = candidates[cell.row]?.[cell.col] ?? []
        if (cellCands.includes(num)) {
          eliminations.push({ row: cell.row, col: cell.col, candidate: num })
          highlights.push({
            row: cell.row,
            col: cell.col,
            type: 'elimination',
            candidates: [num]
          })
        }
      }
    }

    if (eliminations.length > 0) {
      const lineTypeName = lineType === 'row' ? 'row' : 'column'
      const boxNum = Math.floor(firstBox.boxStartRow / boxHeight) * (size / boxWidth) +
                     Math.floor(firstBox.boxStartCol / boxWidth) + 1

      return {
        strategyName: 'Box/Line Reduction',
        difficulty: 1.7,
        description: `${num} in ${lineTypeName} ${lineIndex + 1} is restricted to box ${boxNum}. It can be eliminated from other cells in that box.`,
        eliminations,
        highlights
      }
    }
  }

  return null
}
