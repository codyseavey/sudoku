import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { cellsSeEachOther, getCellsSeeingAll, formatCell } from './utils'

/**
 * WXYZ-Wing (Bent Quad):
 * Four cells containing exactly four candidates {W, X, Y, Z}.
 * One candidate (Z) is the "non-restricted common" - it appears in cells
 * that don't all see each other. The other candidates are "restricted".
 *
 * Any cell that sees ALL instances of Z in the pattern cannot contain Z,
 * because Z must exist somewhere in the pattern.
 */
export const wxyzWing: Strategy = {
  name: 'WXYZ-Wing',
  difficulty: 4.6,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Collect cells with 2-4 candidates
    const eligibleCells: { cell: Cell; cands: number[] }[] = []

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r]?.[c] === null) {
          const cellCands = candidates[r]?.[c] ?? []
          if (cellCands.length >= 2 && cellCands.length <= 4) {
            eligibleCells.push({
              cell: { row: r, col: c },
              cands: [...cellCands]
            })
          }
        }
      }
    }

    // Try all combinations of 4 cells
    for (let i = 0; i < eligibleCells.length; i++) {
      for (let j = i + 1; j < eligibleCells.length; j++) {
        for (let k = j + 1; k < eligibleCells.length; k++) {
          for (let l = k + 1; l < eligibleCells.length; l++) {
            const cells = [
              eligibleCells[i]!,
              eligibleCells[j]!,
              eligibleCells[k]!,
              eligibleCells[l]!
            ]

            // Check if cells form a valid WXYZ-Wing
            const result = checkWXYZWing(cells, board, candidates, size, boxHeight, boxWidth)
            if (result) {
              return result
            }
          }
        }
      }
    }

    return null
  }
}

function checkWXYZWing(
  cells: { cell: Cell; cands: number[] }[],
  board: (number | null)[][],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // Get union of all candidates
  const allCandidates = new Set<number>()
  for (const { cands } of cells) {
    for (const c of cands) {
      allCandidates.add(c)
    }
  }

  // Must have exactly 4 candidates for WXYZ-Wing
  if (allCandidates.size !== 4) {
    return null
  }

  const candidateArray = Array.from(allCandidates)

  // Check connectivity - cells must form a connected pattern
  // At minimum, there must be a "hinge" structure where cells see each other
  if (!isConnectedPattern(cells, boxHeight, boxWidth)) {
    return null
  }

  // Try each candidate as Z (the elimination target)
  for (const Z of candidateArray) {
    // Find all cells containing Z
    const cellsWithZ = cells.filter(c => c.cands.includes(Z))

    if (cellsWithZ.length < 2) continue

    // Z should be "non-restricted" - not all Z cells see each other
    const allZsSeeEachOther = cellsWithZ.every((c1, idx1) =>
      cellsWithZ.every((c2, idx2) =>
        idx1 === idx2 || cellsSeEachOther(c1.cell, c2.cell, boxHeight, boxWidth)
      )
    )

    // For WXYZ-Wing, Z should NOT be fully restricted (otherwise it's a naked quad)
    if (allZsSeeEachOther && cellsWithZ.length === 4) continue

    // Find cells that see ALL instances of Z in the pattern
    const zCells = cellsWithZ.map(c => c.cell)
    const cellsSeeingAllZ = getCellsSeeingAll(zCells, size, boxHeight, boxWidth)

    const eliminations: Elimination[] = []
    const highlights: CellHighlight[] = []

    // Highlight the WXYZ-Wing cells
    for (const wingCell of cells) {
      const hasZ = wingCell.cands.includes(Z)
      highlights.push({
        row: wingCell.cell.row,
        col: wingCell.cell.col,
        type: hasZ ? 'primary' : 'secondary',
        candidates: hasZ ? [Z] : wingCell.cands
      })
    }

    // Find eliminations
    for (const cell of cellsSeeingAllZ) {
      // Skip cells in the pattern
      if (cells.some(c => c.cell.row === cell.row && c.cell.col === cell.col)) {
        continue
      }

      if (board[cell.row]?.[cell.col] === null) {
        const cellCands = candidates[cell.row]?.[cell.col] ?? []
        if (cellCands.includes(Z)) {
          eliminations.push({ row: cell.row, col: cell.col, candidate: Z })
          highlights.push({
            row: cell.row,
            col: cell.col,
            type: 'elimination',
            candidates: [Z]
          })
        }
      }
    }

    if (eliminations.length > 0) {
      const cellNames = cells.map(c => formatCell(c.cell.row, c.cell.col)).join(', ')
      const candNames = candidateArray.join(',')

      return {
        strategyName: 'WXYZ-Wing',
        difficulty: 4.6,
        description: `WXYZ-Wing on {${candNames}} in cells ${cellNames}. The candidate ${Z} must exist in one of these cells, so cells seeing all ${Z}s cannot contain ${Z}.`,
        eliminations,
        highlights
      }
    }
  }

  return null
}

function isConnectedPattern(
  cells: { cell: Cell; cands: number[] }[],
  boxHeight: number,
  boxWidth: number
): boolean {
  // Check if the cells form a connected pattern
  // Each cell should see at least one other cell, and the pattern should be connected
  const visited = new Set<number>()
  const queue: number[] = [0]
  visited.add(0)

  while (queue.length > 0) {
    const current = queue.shift()!
    const currentCell = cells[current]!

    for (let i = 0; i < cells.length; i++) {
      if (!visited.has(i)) {
        if (cellsSeEachOther(currentCell.cell, cells[i]!.cell, boxHeight, boxWidth)) {
          visited.add(i)
          queue.push(i)
        }
      }
    }
  }

  return visited.size === cells.length
}
