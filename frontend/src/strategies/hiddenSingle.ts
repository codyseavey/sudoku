import type { Strategy, SolveContext, HintResult, Cell, CellHighlight } from './types'
import { getAllHouses, formatCell } from './utils'

/**
 * Hidden Single: A candidate that appears in only one cell within a house.
 * Even though the cell may have other candidates, this value must go here.
 */
export const hiddenSingle: Strategy = {
  name: 'Hidden Single',
  difficulty: 1.5,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    const houses = getAllHouses(size, boxHeight, boxWidth)

    for (const house of houses) {
      // For each candidate value 1 to size
      for (let num = 1; num <= size; num++) {
        // Find cells in this house that have this candidate
        const cellsWithCandidate: Cell[] = []

        for (const cell of house.cells) {
          if (board[cell.row]?.[cell.col] === null) {
            const cellCands = candidates[cell.row]?.[cell.col] ?? []
            if (cellCands.includes(num)) {
              cellsWithCandidate.push(cell)
            }
          }
        }

        // If exactly one cell has this candidate, it's a hidden single
        if (cellsWithCandidate.length === 1) {
          const cell = cellsWithCandidate[0]!
          const cellCands = candidates[cell.row]?.[cell.col] ?? []

          // Only report if the cell has multiple candidates (otherwise it's a naked single)
          if (cellCands.length > 1) {
            const houseTypeName = house.type === 'row' ? 'row' :
                                  house.type === 'col' ? 'column' : 'box'
            const houseIndex = house.index + 1

            // The eliminations are the other candidates in this cell
            const eliminations = cellCands
              .filter(c => c !== num)
              .map(c => ({ row: cell.row, col: cell.col, candidate: c }))

            const highlights: CellHighlight[] = [
              {
                row: cell.row,
                col: cell.col,
                type: 'primary',
                candidates: [num]
              }
            ]

            // Add elimination highlights for other candidates
            for (const elim of eliminations) {
              highlights.push({
                row: elim.row,
                col: elim.col,
                type: 'elimination',
                candidates: [elim.candidate]
              })
            }

            return {
              strategyName: 'Hidden Single',
              difficulty: house.type === 'box' ? 1.2 : 1.5,
              description: `${num} can only go in ${formatCell(cell.row, cell.col)} within ${houseTypeName} ${houseIndex}. Other candidates in this cell can be removed.`,
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
