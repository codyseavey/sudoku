import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { cellsSeEachOther, getCellsSeeingAll, formatCell } from './utils'

/**
 * Y-Wing (XY-Wing):
 * Three bi-value cells forming a "bent triple":
 * - Pivot cell with candidates {A, B}
 * - Pincer 1 with candidates {A, C}
 * - Pincer 2 with candidates {B, C}
 * The pivot sees both pincers.
 * Any cell seeing both pincers cannot contain C.
 */
export const yWing: Strategy = {
  name: 'Y-Wing',
  difficulty: 4.2,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Find all bi-value cells (cells with exactly 2 candidates)
    const bivalueCells: { cell: Cell; cands: [number, number] }[] = []

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r]?.[c] === null) {
          const cellCands = candidates[r]?.[c] ?? []
          if (cellCands.length === 2) {
            bivalueCells.push({
              cell: { row: r, col: c },
              cands: [cellCands[0]!, cellCands[1]!]
            })
          }
        }
      }
    }

    // Try each bi-value cell as a potential pivot
    for (const pivot of bivalueCells) {
      const [A, B] = pivot.cands

      // Find potential pincers that the pivot sees
      const potentialPincers = bivalueCells.filter(bc =>
        bc !== pivot &&
        cellsSeEachOther(pivot.cell, bc.cell, boxHeight, boxWidth)
      )

      // Look for pincer1 with {A, C} and pincer2 with {B, C}
      for (const pincer1 of potentialPincers) {
        // Pincer 1 must share exactly one candidate (A or B) with pivot
        const shared1 = pivot.cands.filter(c => pincer1.cands.includes(c))
        if (shared1.length !== 1) continue

        const sharedWithPincer1 = shared1[0]!
        const C_from_pincer1 = pincer1.cands.find(c => c !== sharedWithPincer1)
        if (C_from_pincer1 === undefined) continue

        // The other candidate in pivot must be shared with pincer2
        const otherInPivot = pivot.cands.find(c => c !== sharedWithPincer1)
        if (otherInPivot === undefined) continue

        // Look for pincer2 with {otherInPivot, C_from_pincer1}
        for (const pincer2 of potentialPincers) {
          if (pincer2 === pincer1) continue

          // Pincer 2 must have {otherInPivot, C_from_pincer1}
          if (!pincer2.cands.includes(otherInPivot) ||
              !pincer2.cands.includes(C_from_pincer1)) {
            continue
          }

          // Found a Y-Wing pattern!
          const C = C_from_pincer1

          // Find cells that see both pincers
          const cellsSeeingBothPincers = getCellsSeeingAll(
            [pincer1.cell, pincer2.cell],
            size,
            boxHeight,
            boxWidth
          )

          const eliminations: Elimination[] = []
          const highlights: CellHighlight[] = []

          // Highlight the Y-Wing cells
          highlights.push({
            row: pivot.cell.row,
            col: pivot.cell.col,
            type: 'secondary',
            candidates: pivot.cands
          })
          highlights.push({
            row: pincer1.cell.row,
            col: pincer1.cell.col,
            type: 'primary',
            candidates: [C]
          })
          highlights.push({
            row: pincer2.cell.row,
            col: pincer2.cell.col,
            type: 'primary',
            candidates: [C]
          })

          // Find eliminations
          for (const cell of cellsSeeingBothPincers) {
            // Don't eliminate from the Y-Wing cells themselves
            if ((cell.row === pivot.cell.row && cell.col === pivot.cell.col) ||
                (cell.row === pincer1.cell.row && cell.col === pincer1.cell.col) ||
                (cell.row === pincer2.cell.row && cell.col === pincer2.cell.col)) {
              continue
            }

            if (board[cell.row]?.[cell.col] === null) {
              const cellCands = candidates[cell.row]?.[cell.col] ?? []
              if (cellCands.includes(C)) {
                eliminations.push({ row: cell.row, col: cell.col, candidate: C })
                highlights.push({
                  row: cell.row,
                  col: cell.col,
                  type: 'elimination',
                  candidates: [C]
                })
              }
            }
          }

          if (eliminations.length > 0) {
            return {
              strategyName: 'Y-Wing',
              difficulty: 4.2,
              description: `Y-Wing with pivot ${formatCell(pivot.cell.row, pivot.cell.col)} {${A},${B}} and pincers ${formatCell(pincer1.cell.row, pincer1.cell.col)} and ${formatCell(pincer2.cell.row, pincer2.cell.col)}. Cells seeing both pincers cannot contain ${C}.`,
              eliminations,
              highlights
            }
          }
        }
      }
    }

    return null
  }
}
