import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { cellsSeEachOther, getCellsSeeingAll, formatCell } from './utils'

/**
 * W-Wing:
 * Two bi-value cells containing the exact same pair {A, B} that don't see each other,
 * connected by a "strong link" on one of the candidates (A).
 * A strong link means candidate A appears in exactly 2 cells in a house,
 * and each of those cells sees one of the bi-value cells.
 *
 * Logic: One of the two bi-value cells MUST contain B (because if both were A,
 * the strong link would be violated). Any cell seeing both bi-value cells
 * cannot contain B.
 */
export const wWing: Strategy = {
  name: 'W-Wing',
  difficulty: 4.4,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Find all bi-value cells
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

    // Find pairs of bi-value cells with identical candidates that don't see each other
    for (let i = 0; i < bivalueCells.length; i++) {
      for (let j = i + 1; j < bivalueCells.length; j++) {
        const cell1 = bivalueCells[i]!
        const cell2 = bivalueCells[j]!

        // Must have identical candidates
        if (cell1.cands[0] !== cell2.cands[0] || cell1.cands[1] !== cell2.cands[1]) {
          continue
        }

        // Should NOT see each other (otherwise it's simpler logic)
        if (cellsSeEachOther(cell1.cell, cell2.cell, boxHeight, boxWidth)) {
          continue
        }

        const [A, B] = cell1.cands

        // Try each candidate as the "link" candidate
        for (const linkCandidate of [A, B]) {
          const eliminationCandidate = linkCandidate === A ? B : A

          // Find strong links on linkCandidate that connect to both cells
          const strongLink = findStrongLinkConnecting(
            cell1.cell, cell2.cell, linkCandidate, board, candidates, size, boxHeight, boxWidth
          )

          if (strongLink) {
            // Found a W-Wing! Find eliminations
            const cellsSeeingBoth = getCellsSeeingAll(
              [cell1.cell, cell2.cell],
              size,
              boxHeight,
              boxWidth
            )

            const eliminations: Elimination[] = []
            const highlights: CellHighlight[] = []

            // Highlight the W-Wing cells
            highlights.push({
              row: cell1.cell.row,
              col: cell1.cell.col,
              type: 'primary',
              candidates: [eliminationCandidate]
            })
            highlights.push({
              row: cell2.cell.row,
              col: cell2.cell.col,
              type: 'primary',
              candidates: [eliminationCandidate]
            })

            // Highlight the strong link cells
            highlights.push({
              row: strongLink.cell1.row,
              col: strongLink.cell1.col,
              type: 'secondary',
              candidates: [linkCandidate]
            })
            highlights.push({
              row: strongLink.cell2.row,
              col: strongLink.cell2.col,
              type: 'secondary',
              candidates: [linkCandidate]
            })

            // Find eliminations
            for (const cell of cellsSeeingBoth) {
              if ((cell.row === cell1.cell.row && cell.col === cell1.cell.col) ||
                  (cell.row === cell2.cell.row && cell.col === cell2.cell.col)) {
                continue
              }

              if (board[cell.row]?.[cell.col] === null) {
                const cellCands = candidates[cell.row]?.[cell.col] ?? []
                if (cellCands.includes(eliminationCandidate)) {
                  eliminations.push({ row: cell.row, col: cell.col, candidate: eliminationCandidate })
                  highlights.push({
                    row: cell.row,
                    col: cell.col,
                    type: 'elimination',
                    candidates: [eliminationCandidate]
                  })
                }
              }
            }

            if (eliminations.length > 0) {
              return {
                strategyName: 'W-Wing',
                difficulty: 4.4,
                description: `W-Wing on {${A},${B}}: Cells ${formatCell(cell1.cell.row, cell1.cell.col)} and ${formatCell(cell2.cell.row, cell2.cell.col)} are connected by a strong link on ${linkCandidate}. One must be ${eliminationCandidate}, so cells seeing both cannot contain ${eliminationCandidate}.`,
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

interface StrongLink {
  cell1: Cell
  cell2: Cell
  house: 'row' | 'col' | 'box'
  houseIndex: number
}

function findStrongLinkConnecting(
  endpoint1: Cell,
  endpoint2: Cell,
  candidate: number,
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): StrongLink | null {
  // Check rows for strong links
  for (let r = 0; r < size; r++) {
    const cellsWithCandidate: Cell[] = []
    for (let c = 0; c < size; c++) {
      if (board[r]?.[c] === null) {
        const cellCands = candidates[r]?.[c] ?? []
        if (cellCands.includes(candidate)) {
          cellsWithCandidate.push({ row: r, col: c })
        }
      }
    }

    if (cellsWithCandidate.length === 2) {
      const [link1, link2] = cellsWithCandidate as [Cell, Cell]
      // Check if this strong link connects to both endpoints
      const link1SeesEndpoint1 = cellsSeEachOther(link1, endpoint1, boxHeight, boxWidth)
      const link2SeesEndpoint2 = cellsSeEachOther(link2, endpoint2, boxHeight, boxWidth)
      const link1SeesEndpoint2 = cellsSeEachOther(link1, endpoint2, boxHeight, boxWidth)
      const link2SeesEndpoint1 = cellsSeEachOther(link2, endpoint1, boxHeight, boxWidth)

      if ((link1SeesEndpoint1 && link2SeesEndpoint2) || (link1SeesEndpoint2 && link2SeesEndpoint1)) {
        return { cell1: link1, cell2: link2, house: 'row', houseIndex: r }
      }
    }
  }

  // Check columns for strong links
  for (let c = 0; c < size; c++) {
    const cellsWithCandidate: Cell[] = []
    for (let r = 0; r < size; r++) {
      if (board[r]?.[c] === null) {
        const cellCands = candidates[r]?.[c] ?? []
        if (cellCands.includes(candidate)) {
          cellsWithCandidate.push({ row: r, col: c })
        }
      }
    }

    if (cellsWithCandidate.length === 2) {
      const [link1, link2] = cellsWithCandidate as [Cell, Cell]
      const link1SeesEndpoint1 = cellsSeEachOther(link1, endpoint1, boxHeight, boxWidth)
      const link2SeesEndpoint2 = cellsSeEachOther(link2, endpoint2, boxHeight, boxWidth)
      const link1SeesEndpoint2 = cellsSeEachOther(link1, endpoint2, boxHeight, boxWidth)
      const link2SeesEndpoint1 = cellsSeEachOther(link2, endpoint1, boxHeight, boxWidth)

      if ((link1SeesEndpoint1 && link2SeesEndpoint2) || (link1SeesEndpoint2 && link2SeesEndpoint1)) {
        return { cell1: link1, cell2: link2, house: 'col', houseIndex: c }
      }
    }
  }

  // Check boxes for strong links
  for (let boxRow = 0; boxRow < size / boxHeight; boxRow++) {
    for (let boxCol = 0; boxCol < size / boxWidth; boxCol++) {
      const cellsWithCandidate: Cell[] = []
      const startRow = boxRow * boxHeight
      const startCol = boxCol * boxWidth

      for (let r = startRow; r < startRow + boxHeight; r++) {
        for (let c = startCol; c < startCol + boxWidth; c++) {
          if (board[r]?.[c] === null) {
            const cellCands = candidates[r]?.[c] ?? []
            if (cellCands.includes(candidate)) {
              cellsWithCandidate.push({ row: r, col: c })
            }
          }
        }
      }

      if (cellsWithCandidate.length === 2) {
        const [link1, link2] = cellsWithCandidate as [Cell, Cell]
        const link1SeesEndpoint1 = cellsSeEachOther(link1, endpoint1, boxHeight, boxWidth)
        const link2SeesEndpoint2 = cellsSeEachOther(link2, endpoint2, boxHeight, boxWidth)
        const link1SeesEndpoint2 = cellsSeEachOther(link1, endpoint2, boxHeight, boxWidth)
        const link2SeesEndpoint1 = cellsSeEachOther(link2, endpoint1, boxHeight, boxWidth)

        if ((link1SeesEndpoint1 && link2SeesEndpoint2) || (link1SeesEndpoint2 && link2SeesEndpoint1)) {
          const boxIndex = boxRow * (size / boxWidth) + boxCol
          return { cell1: link1, cell2: link2, house: 'box', houseIndex: boxIndex }
        }
      }
    }
  }

  return null
}
