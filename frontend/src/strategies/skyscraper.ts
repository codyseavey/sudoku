import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getCellsSeeingAll, formatCell } from './utils'

/**
 * Skyscraper:
 * Two rows (or columns) each contain a candidate exactly twice.
 * One pair aligns in a column (the "floor"), the other pair doesn't (the "roof").
 * Any cell that sees both roof cells cannot contain the candidate.
 */
export const skyscraper: Strategy = {
  name: 'Skyscraper',
  difficulty: 4.0,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Try row-based Skyscraper
    const rowResult = findSkyscraper('row', board, candidates, size, boxHeight, boxWidth)
    if (rowResult) return rowResult

    // Try column-based Skyscraper
    const colResult = findSkyscraper('col', board, candidates, size, boxHeight, boxWidth)
    if (colResult) return colResult

    return null
  }
}

function findSkyscraper(
  baseType: 'row' | 'col',
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // For each candidate
  for (let num = 1; num <= size; num++) {
    // Find base lines where this candidate appears in exactly 2 positions
    const linesWithTwoPositions: { line: number; cells: Cell[] }[] = []

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

      if (cells.length === 2) {
        linesWithTwoPositions.push({ line, cells })
      }
    }

    // Look for pairs of lines that form a skyscraper
    for (let i = 0; i < linesWithTwoPositions.length; i++) {
      for (let j = i + 1; j < linesWithTwoPositions.length; j++) {
        const line1 = linesWithTwoPositions[i]!
        const line2 = linesWithTwoPositions[j]!

        // Get the perpendicular positions (columns for row-based, rows for col-based)
        const pos1a = baseType === 'row' ? line1.cells[0]!.col : line1.cells[0]!.row
        const pos1b = baseType === 'row' ? line1.cells[1]!.col : line1.cells[1]!.row
        const pos2a = baseType === 'row' ? line2.cells[0]!.col : line2.cells[0]!.row
        const pos2b = baseType === 'row' ? line2.cells[1]!.col : line2.cells[1]!.row

        // Check for skyscraper pattern: one position matches (floor), one doesn't (roof)
        let floorCell1: Cell | null = null
        let floorCell2: Cell | null = null
        let roofCell1: Cell | null = null
        let roofCell2: Cell | null = null

        if (pos1a === pos2a && pos1b !== pos2b) {
          floorCell1 = line1.cells[0]!
          floorCell2 = line2.cells[0]!
          roofCell1 = line1.cells[1]!
          roofCell2 = line2.cells[1]!
        } else if (pos1a === pos2b && pos1b !== pos2a) {
          floorCell1 = line1.cells[0]!
          floorCell2 = line2.cells[1]!
          roofCell1 = line1.cells[1]!
          roofCell2 = line2.cells[0]!
        } else if (pos1b === pos2a && pos1a !== pos2b) {
          floorCell1 = line1.cells[1]!
          floorCell2 = line2.cells[0]!
          roofCell1 = line1.cells[0]!
          roofCell2 = line2.cells[1]!
        } else if (pos1b === pos2b && pos1a !== pos2a) {
          floorCell1 = line1.cells[1]!
          floorCell2 = line2.cells[1]!
          roofCell1 = line1.cells[0]!
          roofCell2 = line2.cells[0]!
        }

        if (floorCell1 && floorCell2 && roofCell1 && roofCell2) {
          // Found a skyscraper pattern
          // Find cells that see both roof cells
          const cellsSeeingBothRoof = getCellsSeeingAll(
            [roofCell1, roofCell2],
            size,
            boxHeight,
            boxWidth
          )

          const eliminations: Elimination[] = []
          const highlights: CellHighlight[] = []

          // Highlight the skyscraper cells
          highlights.push({ row: floorCell1.row, col: floorCell1.col, type: 'secondary', candidates: [num] })
          highlights.push({ row: floorCell2.row, col: floorCell2.col, type: 'secondary', candidates: [num] })
          highlights.push({ row: roofCell1.row, col: roofCell1.col, type: 'primary', candidates: [num] })
          highlights.push({ row: roofCell2.row, col: roofCell2.col, type: 'primary', candidates: [num] })

          // Find eliminations
          for (const cell of cellsSeeingBothRoof) {
            // Don't eliminate from the skyscraper cells themselves
            if ((cell.row === floorCell1.row && cell.col === floorCell1.col) ||
                (cell.row === floorCell2.row && cell.col === floorCell2.col) ||
                (cell.row === roofCell1.row && cell.col === roofCell1.col) ||
                (cell.row === roofCell2.row && cell.col === roofCell2.col)) {
              continue
            }

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
            const baseTypeName = baseType === 'row' ? 'rows' : 'columns'

            return {
              strategyName: 'Skyscraper',
              difficulty: 4.0,
              description: `Skyscraper on ${num} in ${baseTypeName} ${line1.line + 1} and ${line2.line + 1}. Floor at ${formatCell(floorCell1.row, floorCell1.col)} and ${formatCell(floorCell2.row, floorCell2.col)}, roof at ${formatCell(roofCell1.row, roofCell1.col)} and ${formatCell(roofCell2.row, roofCell2.col)}. Cells seeing both roof cells cannot contain ${num}.`,
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
