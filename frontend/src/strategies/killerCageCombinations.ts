import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cage } from './types'
import { formatCandidates } from './utils'

/**
 * Cage Combinations (Killer Sudoku):
 * Eliminate candidates that cannot be part of any valid combination
 * that sums to the cage total.
 */
export const cageCombinations: Strategy = {
  name: 'Cage Combinations',
  difficulty: 1.5,

  find(context: SolveContext): HintResult | null {
    // Only applies to killer sudoku
    if (context.gameType !== 'killer' || !context.cages) {
      return null
    }

    const { board, candidates, cages, size } = context

    for (const cage of cages) {
      const result = analyzeCage(cage, board, candidates, size)
      if (result) {
        return result
      }
    }

    return null
  }
}

function analyzeCage(
  cage: Cage,
  board: (number | null)[][],
  candidates: number[][][],
  size: number
): HintResult | null {
  // Get empty cells and filled values in the cage
  const emptyCells: { row: number; col: number }[] = []
  const filledValues = new Set<number>()
  let filledSum = 0

  for (const cell of cage.cells) {
    const val = board[cell.row]?.[cell.col]
    if (val === null || val === undefined) {
      emptyCells.push(cell)
    } else {
      filledValues.add(val)
      filledSum += val
    }
  }

  // If cage is complete, nothing to do
  if (emptyCells.length === 0) {
    return null
  }

  const remainingSum = cage.sum - filledSum
  const remainingCells = emptyCells.length

  // Get all possible candidates in the empty cells
  const allCandidatesInCage = new Set<number>()
  for (const cell of emptyCells) {
    const cellCands = candidates[cell.row]?.[cell.col] ?? []
    for (const c of cellCands) {
      allCandidatesInCage.add(c)
    }
  }

  // Find all valid combinations for the remaining cells
  const validCombinations = findCombinations(
    remainingSum,
    remainingCells,
    size,
    filledValues,
    Array.from(allCandidatesInCage)
  )

  if (validCombinations.length === 0) {
    // No valid combinations - puzzle might be invalid
    return null
  }

  // Find candidates that appear in at least one valid combination
  const validCandidates = new Set<number>()
  for (const combo of validCombinations) {
    for (const num of combo) {
      validCandidates.add(num)
    }
  }

  // Find eliminations - candidates not in any valid combination
  const eliminations: Elimination[] = []
  const highlights: CellHighlight[] = []

  for (const cell of emptyCells) {
    const cellCands = candidates[cell.row]?.[cell.col] ?? []
    const invalidCands: number[] = []

    for (const cand of cellCands) {
      if (!validCandidates.has(cand)) {
        eliminations.push({ row: cell.row, col: cell.col, candidate: cand })
        invalidCands.push(cand)
      }
    }

    if (invalidCands.length > 0) {
      highlights.push({
        row: cell.row,
        col: cell.col,
        type: 'elimination',
        candidates: invalidCands
      })
    }
  }

  if (eliminations.length === 0) {
    return null
  }

  // Add cage cells as secondary highlights
  for (const cell of cage.cells) {
    // Don't double-highlight elimination cells
    if (!highlights.some(h => h.row === cell.row && h.col === cell.col)) {
      highlights.push({
        row: cell.row,
        col: cell.col,
        type: 'secondary'
      })
    }
  }

  return {
    strategyName: 'Cage Combinations',
    difficulty: 1.5,
    description: `Cage with sum ${cage.sum} can only use combinations with values ${formatCandidates(Array.from(validCandidates))}. Other candidates can be eliminated.`,
    eliminations,
    highlights
  }
}

/**
 * Find all valid combinations of 'count' distinct digits from availableDigits
 * that sum to 'targetSum', excluding any digits in 'excluded'.
 */
function findCombinations(
  targetSum: number,
  count: number,
  _maxNum: number,
  excluded: Set<number>,
  availableDigits: number[]
): number[][] {
  const results: number[][] = []
  const filtered = availableDigits.filter(d => !excluded.has(d)).sort((a, b) => a - b)

  function backtrack(
    start: number,
    remaining: number,
    currentSum: number,
    current: number[]
  ): void {
    if (remaining === 0) {
      if (currentSum === targetSum) {
        results.push([...current])
      }
      return
    }

    for (let i = start; i < filtered.length; i++) {
      const num = filtered[i]!

      // Skip if exceeds target
      if (currentSum + num > targetSum) continue

      current.push(num)
      backtrack(i + 1, remaining - 1, currentSum + num, current)
      current.pop()
    }
  }

  backtrack(0, count, 0, [])

  return results
}
