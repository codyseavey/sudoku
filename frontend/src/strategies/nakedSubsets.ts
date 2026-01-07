import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getAllHouses, formatCell, formatCandidates } from './utils'

/**
 * Naked Subsets (Pairs, Triples, Quads):
 * If N cells in a house contain only N candidates total (distributed among them),
 * those candidates can be eliminated from other cells in the house.
 */
export const nakedSubsets: Strategy = {
  name: 'Naked Subsets',
  difficulty: 2.5,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    const houses = getAllHouses(size, boxHeight, boxWidth)

    // Try to find naked pairs first (easiest to spot), then triples, then quads
    for (const subsetSize of [2, 3, 4]) {
      for (const house of houses) {
        const result = findNakedSubset(house, subsetSize, board, candidates)
        if (result) return result
      }
    }

    return null
  }
}

function findNakedSubset(
  house: { type: 'row' | 'col' | 'box'; index: number; cells: Cell[] },
  subsetSize: number,
  board: (number | null)[][],
  candidates: number[][][]
): HintResult | null {
  // Get empty cells with their candidates
  const emptyCells: { cell: Cell; cands: number[] }[] = []

  for (const cell of house.cells) {
    if (board[cell.row]?.[cell.col] === null) {
      const cands = candidates[cell.row]?.[cell.col] ?? []
      if (cands.length >= 2 && cands.length <= subsetSize) {
        emptyCells.push({ cell, cands })
      }
    }
  }

  if (emptyCells.length < subsetSize) return null

  // Try all combinations of 'subsetSize' cells
  const combinations = getCombinations(emptyCells, subsetSize)

  for (const combo of combinations) {
    // Get union of all candidates in this combination
    const unionCands = new Set<number>()
    for (const { cands } of combo) {
      for (const c of cands) {
        unionCands.add(c)
      }
    }

    // If the union has exactly 'subsetSize' candidates, we have a naked subset
    if (unionCands.size === subsetSize) {
      const subsetCells = combo.map(c => c.cell)
      const subsetCandidates = Array.from(unionCands)

      // Find eliminations in other cells of the house
      const eliminations: Elimination[] = []
      const highlights: CellHighlight[] = []

      // Highlight the naked subset cells
      for (const { cell, cands } of combo) {
        highlights.push({
          row: cell.row,
          col: cell.col,
          type: 'primary',
          candidates: cands
        })
      }

      // Find eliminations
      for (const cell of house.cells) {
        // Skip cells in the subset
        if (subsetCells.some(sc => sc.row === cell.row && sc.col === cell.col)) {
          continue
        }

        if (board[cell.row]?.[cell.col] === null) {
          const cellCands = candidates[cell.row]?.[cell.col] ?? []
          const toEliminate = cellCands.filter(c => subsetCandidates.includes(c))

          for (const c of toEliminate) {
            eliminations.push({ row: cell.row, col: cell.col, candidate: c })
          }

          if (toEliminate.length > 0) {
            highlights.push({
              row: cell.row,
              col: cell.col,
              type: 'elimination',
              candidates: toEliminate
            })
          }
        }
      }

      if (eliminations.length > 0) {
        const subsetName = subsetSize === 2 ? 'Naked Pair' :
                          subsetSize === 3 ? 'Naked Triple' : 'Naked Quad'
        const difficulty = subsetSize === 2 ? 2.0 :
                          subsetSize === 3 ? 2.8 : 3.0
        const houseTypeName = house.type === 'row' ? 'row' :
                              house.type === 'col' ? 'column' : 'box'

        return {
          strategyName: subsetName,
          difficulty,
          description: `${subsetName} ${formatCandidates(subsetCandidates)} found in ${houseTypeName} ${house.index + 1} at ${subsetCells.map(c => formatCell(c.row, c.col)).join(', ')}. These candidates can be eliminated from other cells.`,
          eliminations,
          highlights
        }
      }
    }
  }

  return null
}

function getCombinations<T>(arr: T[], size: number): T[][] {
  if (size === 0) return [[]]
  if (arr.length < size) return []

  const result: T[][] = []

  function backtrack(start: number, current: T[]): void {
    if (current.length === size) {
      result.push([...current])
      return
    }

    for (let i = start; i < arr.length; i++) {
      current.push(arr[i]!)
      backtrack(i + 1, current)
      current.pop()
    }
  }

  backtrack(0, [])
  return result
}
