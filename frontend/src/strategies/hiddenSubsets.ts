import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getAllHouses, formatCell, formatCandidates } from './utils'

/**
 * Hidden Subsets (Pairs, Triples, Quads):
 * If N candidates appear in only N cells within a house,
 * other candidates can be eliminated from those N cells.
 */
export const hiddenSubsets: Strategy = {
  name: 'Hidden Subsets',
  difficulty: 2.8,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    const houses = getAllHouses(size, boxHeight, boxWidth)

    // Try to find hidden pairs first, then triples, then quads
    for (const subsetSize of [2, 3, 4]) {
      for (const house of houses) {
        const result = findHiddenSubset(house, subsetSize, board, candidates)
        if (result) return result
      }
    }

    return null
  }
}

function findHiddenSubset(
  house: { type: 'row' | 'col' | 'box'; index: number; cells: Cell[] },
  subsetSize: number,
  board: (number | null)[][],
  candidates: number[][][]
): HintResult | null {
  // Build a map of candidate -> cells that contain it
  const candidateLocations = new Map<number, Cell[]>()

  for (const cell of house.cells) {
    if (board[cell.row]?.[cell.col] === null) {
      const cands = candidates[cell.row]?.[cell.col] ?? []
      for (const c of cands) {
        if (!candidateLocations.has(c)) {
          candidateLocations.set(c, [])
        }
        candidateLocations.get(c)!.push(cell)
      }
    }
  }

  // Find candidates that appear in 2 to subsetSize cells
  const eligibleCandidates: number[] = []
  for (const [cand, cells] of candidateLocations) {
    if (cells.length >= 2 && cells.length <= subsetSize) {
      eligibleCandidates.push(cand)
    }
  }

  if (eligibleCandidates.length < subsetSize) return null

  // Try all combinations of 'subsetSize' candidates
  const combinations = getCombinations(eligibleCandidates, subsetSize)

  for (const candCombo of combinations) {
    // Get union of all cells that contain any of these candidates
    const cellSet = new Set<string>()
    const cells: Cell[] = []

    for (const cand of candCombo) {
      const locations = candidateLocations.get(cand) ?? []
      for (const cell of locations) {
        const key = `${cell.row},${cell.col}`
        if (!cellSet.has(key)) {
          cellSet.add(key)
          cells.push(cell)
        }
      }
    }

    // If these N candidates appear in exactly N cells, we have a hidden subset
    if (cells.length === subsetSize) {
      // Find eliminations - other candidates in these cells
      const eliminations: Elimination[] = []
      const highlights: CellHighlight[] = []

      for (const cell of cells) {
        const cellCands = candidates[cell.row]?.[cell.col] ?? []
        const toKeep = cellCands.filter(c => candCombo.includes(c))
        const toEliminate = cellCands.filter(c => !candCombo.includes(c))

        // Highlight the hidden subset candidates
        highlights.push({
          row: cell.row,
          col: cell.col,
          type: 'primary',
          candidates: toKeep
        })

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

      if (eliminations.length > 0) {
        const subsetName = subsetSize === 2 ? 'Hidden Pair' :
                          subsetSize === 3 ? 'Hidden Triple' : 'Hidden Quad'
        const difficulty = subsetSize === 2 ? 2.3 :
                          subsetSize === 3 ? 3.0 : 3.2
        const houseTypeName = house.type === 'row' ? 'row' :
                              house.type === 'col' ? 'column' : 'box'

        return {
          strategyName: subsetName,
          difficulty,
          description: `${subsetName} ${formatCandidates(candCombo)} found in ${houseTypeName} ${house.index + 1}. These candidates only appear in ${cells.map(c => formatCell(c.row, c.col)).join(', ')}, so other candidates can be removed from these cells.`,
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
