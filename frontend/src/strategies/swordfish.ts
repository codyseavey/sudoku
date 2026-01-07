import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight } from './types'

/**
 * Swordfish:
 * An extension of X-Wing to 3 rows/columns.
 * If a candidate appears in 2-3 positions in 3 rows,
 * and all positions fall within 3 columns,
 * the candidate can be eliminated from other cells in those columns.
 */
export const swordfish: Strategy = {
  name: 'Swordfish',
  difficulty: 3.8,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size } = context

    // Try row-based Swordfish first
    const rowResult = findSwordfish('row', board, candidates, size)
    if (rowResult) return rowResult

    // Try column-based Swordfish
    const colResult = findSwordfish('col', board, candidates, size)
    if (colResult) return colResult

    return null
  }
}

function findSwordfish(
  baseType: 'row' | 'col',
  board: (number | null)[][],
  candidates: number[][][],
  size: number
): HintResult | null {
  // For each candidate
  for (let num = 1; num <= size; num++) {
    // Find base lines where this candidate appears in 2-3 positions
    const eligibleLines: { line: number; positions: number[] }[] = []

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

      if (positions.length >= 2 && positions.length <= 3) {
        eligibleLines.push({ line, positions })
      }
    }

    if (eligibleLines.length < 3) continue

    // Try all combinations of 3 lines
    for (let i = 0; i < eligibleLines.length; i++) {
      for (let j = i + 1; j < eligibleLines.length; j++) {
        for (let k = j + 1; k < eligibleLines.length; k++) {
          const line1 = eligibleLines[i]!
          const line2 = eligibleLines[j]!
          const line3 = eligibleLines[k]!

          // Get union of all positions
          const allPositions = new Set<number>([
            ...line1.positions,
            ...line2.positions,
            ...line3.positions
          ])

          // If all positions fall within exactly 3 columns, we have a Swordfish
          if (allPositions.size === 3) {
            const coverPositions = Array.from(allPositions)

            // Find eliminations in the cover lines
            const eliminations: Elimination[] = []
            const highlights: CellHighlight[] = []

            // Highlight the Swordfish cells
            const lines = [line1, line2, line3]
            for (const lineData of lines) {
              for (const pos of lineData.positions) {
                const row = baseType === 'row' ? lineData.line : pos
                const col = baseType === 'row' ? pos : lineData.line
                highlights.push({
                  row,
                  col,
                  type: 'primary',
                  candidates: [num]
                })
              }
            }

            // Eliminate from cover positions
            for (const coverPos of coverPositions) {
              for (let lineIdx = 0; lineIdx < size; lineIdx++) {
                // Skip the base lines
                if (lineIdx === line1.line || lineIdx === line2.line || lineIdx === line3.line) {
                  continue
                }

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
                strategyName: 'Swordfish',
                difficulty: 3.8,
                description: `Swordfish on ${num} in ${baseTypeName} ${line1.line + 1}, ${line2.line + 1}, and ${line3.line + 1}, covering ${coverTypeName} ${coverPositions.map(p => p + 1).join(', ')}. The candidate can be eliminated from other cells in those ${coverTypeName}.`,
                eliminations,
                highlights
              }
            }
          }
        }
      }
    }
  }

  return null
}
