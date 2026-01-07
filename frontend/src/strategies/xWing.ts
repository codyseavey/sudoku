import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight } from './types'

/**
 * X-Wing:
 * If a candidate appears in exactly 2 positions in 2 rows,
 * and these positions align in 2 columns (forming a rectangle),
 * the candidate can be eliminated from other cells in those columns.
 * Also works with columns as the base and rows as the cover.
 */
export const xWing: Strategy = {
  name: 'X-Wing',
  difficulty: 3.2,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size } = context

    // Try row-based X-Wing first
    const rowResult = findXWing('row', board, candidates, size)
    if (rowResult) return rowResult

    // Try column-based X-Wing
    const colResult = findXWing('col', board, candidates, size)
    if (colResult) return colResult

    return null
  }
}

function findXWing(
  baseType: 'row' | 'col',
  board: (number | null)[][],
  candidates: number[][][],
  size: number
): HintResult | null {
  // For each candidate
  for (let num = 1; num <= size; num++) {
    // Find base lines where this candidate appears in exactly 2 positions
    const linesWithTwoPositions: { line: number; positions: number[] }[] = []

    for (let line = 0; line < size; line++) {
      const positions: number[] = []

      for (let pos = 0; pos < size; pos++) {
        const row = baseType === 'row' ? line : pos
        const col = baseType === 'row' ? pos : line

        if (board[row]?.[col] === null) {
          const cellCands = candidates[row]?.[col] ?? []
          if (cellCands.includes(num)) {
            positions.push(pos)
          }
        }
      }

      if (positions.length === 2) {
        linesWithTwoPositions.push({ line, positions })
      }
    }

    // Look for pairs of lines with matching positions
    for (let i = 0; i < linesWithTwoPositions.length; i++) {
      for (let j = i + 1; j < linesWithTwoPositions.length; j++) {
        const line1 = linesWithTwoPositions[i]!
        const line2 = linesWithTwoPositions[j]!

        // Check if positions match
        if (line1.positions[0] === line2.positions[0] &&
            line1.positions[1] === line2.positions[1]) {
          // Found an X-Wing pattern
          const pos1 = line1.positions[0]!
          const pos2 = line1.positions[1]!

          // Find eliminations in the cover lines (perpendicular)
          const eliminations: Elimination[] = []
          const highlights: CellHighlight[] = []

          // Highlight the X-Wing cells
          const xWingCells = [
            baseType === 'row' ? { row: line1.line, col: pos1 } : { row: pos1, col: line1.line },
            baseType === 'row' ? { row: line1.line, col: pos2 } : { row: pos2, col: line1.line },
            baseType === 'row' ? { row: line2.line, col: pos1 } : { row: pos1, col: line2.line },
            baseType === 'row' ? { row: line2.line, col: pos2 } : { row: pos2, col: line2.line },
          ]

          for (const cell of xWingCells) {
            highlights.push({
              row: cell.row,
              col: cell.col,
              type: 'primary',
              candidates: [num]
            })
          }

          // Eliminate from cover lines
          for (const coverPos of [pos1, pos2]) {
            for (let lineIdx = 0; lineIdx < size; lineIdx++) {
              // Skip the base lines
              if (lineIdx === line1.line || lineIdx === line2.line) continue

              const row = baseType === 'row' ? lineIdx : coverPos
              const col = baseType === 'row' ? coverPos : lineIdx

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
            const coverTypeName = baseType === 'row' ? 'columns' : 'rows'

            return {
              strategyName: 'X-Wing',
              difficulty: 3.2,
              description: `X-Wing on ${num} in ${baseTypeName} ${line1.line + 1} and ${line2.line + 1}, ${coverTypeName} ${pos1 + 1} and ${pos2 + 1}. The candidate can be eliminated from other cells in those ${coverTypeName}.`,
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
