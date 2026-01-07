import type { Strategy, SolveContext, HintResult } from './types'
import { formatCell } from './utils'

/**
 * Naked Single: A cell with only one candidate.
 * This is technically a placement hint, not an elimination hint,
 * but it's useful to show users when a cell can be solved.
 */
export const nakedSingle: Strategy = {
  name: 'Naked Single',
  difficulty: 1.0,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size } = context

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        // Skip filled cells
        if (board[r]?.[c] !== null) continue

        const cellCandidates = candidates[r]?.[c] ?? []

        // Found a naked single
        if (cellCandidates.length === 1) {
          const value = cellCandidates[0]!
          return {
            strategyName: 'Naked Single',
            difficulty: 1.0,
            description: `${formatCell(r, c)} can only be ${value} - it's the only candidate remaining in this cell.`,
            eliminations: [], // No eliminations - this is a placement
            highlights: [
              {
                row: r,
                col: c,
                type: 'primary',
                candidates: [value]
              }
            ]
          }
        }
      }
    }

    return null
  }
}
