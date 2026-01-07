import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { cellsSeEachOther, formatCell } from './utils'

/**
 * XYZ-Wing:
 * Similar to Y-Wing but the pivot has 3 candidates:
 * - Pivot cell with candidates {X, Y, Z}
 * - Pincer 1 with candidates {X, Z}
 * - Pincer 2 with candidates {Y, Z}
 * The pivot sees both pincers.
 * Any cell seeing ALL THREE (pivot + both pincers) cannot contain Z.
 */
export const xyzWing: Strategy = {
  name: 'XYZ-Wing',
  difficulty: 4.4,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Find all tri-value cells (potential pivots)
    const trivalueCells: { cell: Cell; cands: number[] }[] = []
    // Find all bi-value cells (potential pincers)
    const bivalueCells: { cell: Cell; cands: [number, number] }[] = []

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r]?.[c] === null) {
          const cellCands = candidates[r]?.[c] ?? []
          if (cellCands.length === 3) {
            trivalueCells.push({
              cell: { row: r, col: c },
              cands: [...cellCands]
            })
          } else if (cellCands.length === 2) {
            bivalueCells.push({
              cell: { row: r, col: c },
              cands: [cellCands[0]!, cellCands[1]!]
            })
          }
        }
      }
    }

    // Try each tri-value cell as a potential pivot
    for (const pivot of trivalueCells) {
      // Try each candidate as Z (the one that gets eliminated)
      for (const Z of pivot.cands) {
        const others = pivot.cands.filter(c => c !== Z)
        if (others.length !== 2) continue

        const X = others[0]!
        const Y = others[1]!

        // Find pincers that the pivot sees
        const visiblePincers = bivalueCells.filter(bc =>
          cellsSeEachOther(pivot.cell, bc.cell, boxHeight, boxWidth)
        )

        // Look for pincer1 with {X, Z} and pincer2 with {Y, Z}
        for (const pincer1 of visiblePincers) {
          // Pincer 1 must have exactly {X, Z}
          if (!(pincer1.cands.includes(X) && pincer1.cands.includes(Z) &&
                !pincer1.cands.includes(Y))) {
            continue
          }

          for (const pincer2 of visiblePincers) {
            if (pincer2 === pincer1) continue

            // Pincer 2 must have exactly {Y, Z}
            if (!(pincer2.cands.includes(Y) && pincer2.cands.includes(Z) &&
                  !pincer2.cands.includes(X))) {
              continue
            }

            // Found an XYZ-Wing pattern!
            // Find cells that see all three (pivot + both pincers)
            const eliminations: Elimination[] = []
            const highlights: CellHighlight[] = []

            // Highlight the XYZ-Wing cells
            highlights.push({
              row: pivot.cell.row,
              col: pivot.cell.col,
              type: 'primary',
              candidates: [Z]
            })
            highlights.push({
              row: pincer1.cell.row,
              col: pincer1.cell.col,
              type: 'primary',
              candidates: [Z]
            })
            highlights.push({
              row: pincer2.cell.row,
              col: pincer2.cell.col,
              type: 'primary',
              candidates: [Z]
            })

            // Find cells that see all three XYZ-Wing cells
            for (let r = 0; r < size; r++) {
              for (let c = 0; c < size; c++) {
                const testCell: Cell = { row: r, col: c }

                // Skip the XYZ-Wing cells themselves
                if ((r === pivot.cell.row && c === pivot.cell.col) ||
                    (r === pincer1.cell.row && c === pincer1.cell.col) ||
                    (r === pincer2.cell.row && c === pincer2.cell.col)) {
                  continue
                }

                // Must see all three
                if (cellsSeEachOther(testCell, pivot.cell, boxHeight, boxWidth) &&
                    cellsSeEachOther(testCell, pincer1.cell, boxHeight, boxWidth) &&
                    cellsSeEachOther(testCell, pincer2.cell, boxHeight, boxWidth)) {

                  if (board[r]?.[c] === null) {
                    const cellCands = candidates[r]?.[c] ?? []
                    if (cellCands.includes(Z)) {
                      eliminations.push({ row: r, col: c, candidate: Z })
                      highlights.push({
                        row: r,
                        col: c,
                        type: 'elimination',
                        candidates: [Z]
                      })
                    }
                  }
                }
              }
            }

            if (eliminations.length > 0) {
              return {
                strategyName: 'XYZ-Wing',
                difficulty: 4.4,
                description: `XYZ-Wing with pivot ${formatCell(pivot.cell.row, pivot.cell.col)} {${X},${Y},${Z}} and pincers ${formatCell(pincer1.cell.row, pincer1.cell.col)} {${X},${Z}} and ${formatCell(pincer2.cell.row, pincer2.cell.col)} {${Y},${Z}}. Cells seeing all three cannot contain ${Z}.`,
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
