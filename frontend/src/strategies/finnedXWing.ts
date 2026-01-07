import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getBoxIndex, formatCell } from './utils'

/**
 * Finned X-Wing:
 * An X-Wing pattern with extra candidates ("fins") in one of the base rows/columns.
 *
 * Logic: Either the standard X-Wing holds (fin is false), or the fin is true.
 * Eliminations valid in BOTH scenarios can be made - typically cells that are
 * both targeted by the X-Wing AND see the fin.
 */
export const finnedXWing: Strategy = {
  name: 'Finned X-Wing',
  difficulty: 3.4,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Try row-based finned X-Wing
    const rowResult = findFinnedXWing('row', board, candidates, size, boxHeight, boxWidth)
    if (rowResult) return rowResult

    // Try column-based finned X-Wing
    const colResult = findFinnedXWing('col', board, candidates, size, boxHeight, boxWidth)
    if (colResult) return colResult

    return null
  }
}

function findFinnedXWing(
  baseType: 'row' | 'col',
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // For each candidate
  for (let num = 1; num <= size; num++) {
    // Find base lines where candidate appears in 2 positions (perfect) or 2-3 positions (potential fin)
    const lineData: { line: number; cells: Cell[] }[] = []

    for (let line = 0; line < size; line++) {
      const cells: Cell[] = []

      for (let pos = 0; pos < size; pos++) {
        const row = baseType === 'row' ? line : pos
        const col = baseType === 'row' ? pos : line

        if (board[row]?.[col] === null) {
          const cellCands = candidates[row]?.[col] ?? []
          if (cellCands.includes(num)) {
            cells.push({ row, col })
          }
        }
      }

      if (cells.length >= 2 && cells.length <= 4) {
        lineData.push({ line, cells })
      }
    }

    // Look for finned X-Wing patterns
    for (let i = 0; i < lineData.length; i++) {
      for (let j = i + 1; j < lineData.length; j++) {
        const line1 = lineData[i]!
        const line2 = lineData[j]!

        // One line should have exactly 2 positions, other can have 2-4 (with fins)
        const result = checkFinnedPattern(
          line1, line2, num, baseType, board, candidates, size, boxHeight, boxWidth
        )
        if (result) return result

        // Try reversed
        const resultReversed = checkFinnedPattern(
          line2, line1, num, baseType, board, candidates, size, boxHeight, boxWidth
        )
        if (resultReversed) return resultReversed
      }
    }
  }

  return null
}

function checkFinnedPattern(
  perfectLine: { line: number; cells: Cell[] },
  finnedLine: { line: number; cells: Cell[] },
  num: number,
  baseType: 'row' | 'col',
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // Perfect line should have exactly 2 cells
  if (perfectLine.cells.length !== 2) return null

  // Finned line should have 3-4 cells (2 core + 1-2 fins)
  if (finnedLine.cells.length < 3 || finnedLine.cells.length > 4) return null

  // Get positions in cover dimension
  const perfectPos1 = baseType === 'row' ? perfectLine.cells[0]!.col : perfectLine.cells[0]!.row
  const perfectPos2 = baseType === 'row' ? perfectLine.cells[1]!.col : perfectLine.cells[1]!.row

  // Find which cells in finned line match the perfect positions
  const coreCells: Cell[] = []
  const finCells: Cell[] = []

  for (const cell of finnedLine.cells) {
    const pos = baseType === 'row' ? cell.col : cell.row
    if (pos === perfectPos1 || pos === perfectPos2) {
      coreCells.push(cell)
    } else {
      finCells.push(cell)
    }
  }

  // Must have exactly 2 core cells and 1-2 fins
  if (coreCells.length !== 2 || finCells.length < 1 || finCells.length > 2) {
    return null
  }

  // All fins must be in the same box
  const finBoxes = finCells.map(c => getBoxIndex(c.row, c.col, boxHeight, boxWidth))
  if (finBoxes.length > 1 && finBoxes[0] !== finBoxes[1]) {
    return null
  }
  const finBox = finBoxes[0]!

  // At least one core cell in the finned line must be in the same box as the fins
  const coreInFinBox = coreCells.filter(c =>
    getBoxIndex(c.row, c.col, boxHeight, boxWidth) === finBox
  )
  if (coreInFinBox.length === 0) {
    return null
  }

  // Find eliminations: cells that are both in the cover lines AND in the fin's box
  const eliminations: Elimination[] = []
  const highlights: CellHighlight[] = []

  // Highlight the X-Wing cells
  for (const cell of perfectLine.cells) {
    highlights.push({
      row: cell.row,
      col: cell.col,
      type: 'primary',
      candidates: [num]
    })
  }
  for (const cell of coreCells) {
    highlights.push({
      row: cell.row,
      col: cell.col,
      type: 'primary',
      candidates: [num]
    })
  }

  // Highlight the fins
  for (const cell of finCells) {
    highlights.push({
      row: cell.row,
      col: cell.col,
      type: 'secondary',
      candidates: [num]
    })
  }

  // Check cells in the cover lines that are also in the fin's box
  for (const coverPos of [perfectPos1, perfectPos2]) {
    for (let lineIdx = 0; lineIdx < size; lineIdx++) {
      // Skip the base lines
      if (lineIdx === perfectLine.line || lineIdx === finnedLine.line) continue

      const row = baseType === 'row' ? lineIdx : coverPos
      const col = baseType === 'row' ? coverPos : lineIdx

      // Must be in the fin's box
      if (getBoxIndex(row, col, boxHeight, boxWidth) !== finBox) continue

      if (board[row]?.[col] === null) {
        const cellCands = candidates[row]?.[col] ?? []
        if (cellCands.includes(num)) {
          eliminations.push({ row, col, candidate: num })
          highlights.push({
            row,
            col,
            type: 'elimination',
            candidates: [num]
          })
        }
      }
    }
  }

  if (eliminations.length > 0) {
    const baseTypeName = baseType === 'row' ? 'rows' : 'columns'
    const finCellNames = finCells.map(c => formatCell(c.row, c.col)).join(', ')

    return {
      strategyName: 'Finned X-Wing',
      difficulty: 3.4,
      description: `Finned X-Wing on ${num} in ${baseTypeName} ${perfectLine.line + 1} and ${finnedLine.line + 1}. Fin at ${finCellNames}. Cells in both the cover lines and the fin's box cannot contain ${num}.`,
      eliminations,
      highlights
    }
  }

  return null
}
