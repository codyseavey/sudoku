import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell } from './types'
import { getBoxIndex, formatCell } from './utils'

/**
 * Unique Rectangle:
 * Exploits the uniqueness constraint - a valid Sudoku has exactly one solution.
 * A "deadly pattern" of 4 cells forming a rectangle (2 rows, 2 cols, 2 boxes)
 * with only candidates {A, B} would create two valid solutions.
 * We can make eliminations to prevent this pattern from forming.
 *
 * Type 1: Three corners have only {A,B}, fourth has {A,B,X} - fourth must be X
 * Type 2: Two corners have {A,B}, two have {A,B,X} in same house - X eliminates from house
 */
export const uniqueRectangle: Strategy = {
  name: 'Unique Rectangle',
  difficulty: 4.5,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, boxHeight, boxWidth } = context

    // Find cells with exactly 2 candidates
    const bivalueCells: { cell: Cell; cands: [number, number] }[] = []
    // Find cells with 3+ candidates containing potential UR pairs
    const multivalueCells: { cell: Cell; cands: number[] }[] = []

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r]?.[c] === null) {
          const cellCands = candidates[r]?.[c] ?? []
          if (cellCands.length === 2) {
            bivalueCells.push({
              cell: { row: r, col: c },
              cands: [cellCands[0]!, cellCands[1]!]
            })
          } else if (cellCands.length >= 3) {
            multivalueCells.push({
              cell: { row: r, col: c },
              cands: [...cellCands]
            })
          }
        }
      }
    }

    // Try Type 1: Three corners with {A,B}, one corner with {A,B,X...}
    const type1Result = findType1(bivalueCells, multivalueCells, boxHeight, boxWidth)
    if (type1Result) return type1Result

    // Try Type 2: Two corners with {A,B}, two corners with {A,B,X}
    const type2Result = findType2(bivalueCells, multivalueCells, candidates, size, boxHeight, boxWidth)
    if (type2Result) return type2Result

    return null
  }
}

function findType1(
  bivalueCells: { cell: Cell; cands: [number, number] }[],
  multivalueCells: { cell: Cell; cands: number[] }[],
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // Find 3 bivalue cells that could form 3 corners of a rectangle
  for (let i = 0; i < bivalueCells.length; i++) {
    for (let j = i + 1; j < bivalueCells.length; j++) {
      for (let k = j + 1; k < bivalueCells.length; k++) {
        const c1 = bivalueCells[i]!
        const c2 = bivalueCells[j]!
        const c3 = bivalueCells[k]!

        // All three must have the same candidates
        if (c1.cands[0] !== c2.cands[0] || c1.cands[1] !== c2.cands[1] ||
            c1.cands[0] !== c3.cands[0] || c1.cands[1] !== c3.cands[1]) {
          continue
        }

        const [A, B] = c1.cands

        // Check if they form 3 corners of a rectangle
        const rows = new Set([c1.cell.row, c2.cell.row, c3.cell.row])
        const cols = new Set([c1.cell.col, c2.cell.col, c3.cell.col])

        if (rows.size !== 2 || cols.size !== 2) continue

        const rowArr = Array.from(rows)
        const colArr = Array.from(cols)

        // Find the missing corner
        const corners = [
          { row: rowArr[0]!, col: colArr[0]! },
          { row: rowArr[0]!, col: colArr[1]! },
          { row: rowArr[1]!, col: colArr[0]! },
          { row: rowArr[1]!, col: colArr[1]! }
        ]

        const existingCorners = [c1.cell, c2.cell, c3.cell]
        const missingCorner = corners.find(corner =>
          !existingCorners.some(ec => ec.row === corner.row && ec.col === corner.col)
        )

        if (!missingCorner) continue

        // Rectangle must span exactly 2 boxes
        const boxes = new Set(corners.map(c => getBoxIndex(c.row, c.col, boxHeight, boxWidth)))
        if (boxes.size !== 2) continue

        // Check if missing corner has {A, B, X...}
        const fourthCell = multivalueCells.find(mc =>
          mc.cell.row === missingCorner.row &&
          mc.cell.col === missingCorner.col &&
          mc.cands.includes(A) &&
          mc.cands.includes(B)
        )

        if (!fourthCell) continue

        // Found Type 1 UR! The fourth cell must NOT be A or B
        const extraCandidates = fourthCell.cands.filter(c => c !== A && c !== B)

        if (extraCandidates.length === 1) {
          // Can place the extra candidate directly
          const X = extraCandidates[0]!
          const eliminations: Elimination[] = [
            { row: fourthCell.cell.row, col: fourthCell.cell.col, candidate: A },
            { row: fourthCell.cell.row, col: fourthCell.cell.col, candidate: B }
          ]

          const highlights: CellHighlight[] = [
            { row: c1.cell.row, col: c1.cell.col, type: 'secondary', candidates: [A, B] },
            { row: c2.cell.row, col: c2.cell.col, type: 'secondary', candidates: [A, B] },
            { row: c3.cell.row, col: c3.cell.col, type: 'secondary', candidates: [A, B] },
            { row: fourthCell.cell.row, col: fourthCell.cell.col, type: 'primary', candidates: [X] },
            { row: fourthCell.cell.row, col: fourthCell.cell.col, type: 'elimination', candidates: [A, B] }
          ]

          return {
            strategyName: 'Unique Rectangle Type 1',
            difficulty: 4.5,
            description: `Unique Rectangle on {${A},${B}}. Three corners have only these candidates. To avoid a deadly pattern (multiple solutions), ${formatCell(fourthCell.cell.row, fourthCell.cell.col)} must be ${X}.`,
            eliminations,
            highlights
          }
        }
      }
    }
  }

  return null
}

function findType2(
  bivalueCells: { cell: Cell; cands: [number, number] }[],
  multivalueCells: { cell: Cell; cands: number[] }[],
  candidates: number[][][],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // Find pairs of bivalue cells that could form 2 diagonal corners
  for (let i = 0; i < bivalueCells.length; i++) {
    for (let j = i + 1; j < bivalueCells.length; j++) {
      const c1 = bivalueCells[i]!
      const c2 = bivalueCells[j]!

      // Must have same candidates
      if (c1.cands[0] !== c2.cands[0] || c1.cands[1] !== c2.cands[1]) continue

      // Must be diagonal (different row AND different col)
      if (c1.cell.row === c2.cell.row || c1.cell.col === c2.cell.col) continue

      const [A, B] = c1.cands

      // Find the other two corners
      const corner3: Cell = { row: c1.cell.row, col: c2.cell.col }
      const corner4: Cell = { row: c2.cell.row, col: c1.cell.col }

      // Both must exist and contain A, B, and at least one extra
      const cell3 = multivalueCells.find(mc =>
        mc.cell.row === corner3.row &&
        mc.cell.col === corner3.col &&
        mc.cands.includes(A) &&
        mc.cands.includes(B)
      )

      const cell4 = multivalueCells.find(mc =>
        mc.cell.row === corner4.row &&
        mc.cell.col === corner4.col &&
        mc.cands.includes(A) &&
        mc.cands.includes(B)
      )

      if (!cell3 || !cell4) continue

      // Rectangle must span exactly 2 boxes
      const corners = [c1.cell, c2.cell, corner3, corner4]
      const boxes = new Set(corners.map(c => getBoxIndex(c.row, c.col, boxHeight, boxWidth)))
      if (boxes.size !== 2) continue

      // Find common extra candidates
      const extras3 = cell3.cands.filter(c => c !== A && c !== B)
      const extras4 = cell4.cands.filter(c => c !== A && c !== B)
      const commonExtras = extras3.filter(e => extras4.includes(e))

      if (commonExtras.length === 0) continue

      // Type 2: If both extra corners have the same single extra candidate X,
      // and they're in the same house, X can be eliminated from that house
      if (extras3.length === 1 && extras4.length === 1 && extras3[0] === extras4[0]) {
        const X = extras3[0]!

        // Check if they share a house (row, col, or box)
        let sharedHouseCells: Cell[] = []
        let houseType = ''

        if (corner3.row === corner4.row) {
          // Same row
          houseType = `row ${corner3.row + 1}`
          for (let c = 0; c < size; c++) {
            if (c !== corner3.col && c !== corner4.col) {
              sharedHouseCells.push({ row: corner3.row, col: c })
            }
          }
        } else if (corner3.col === corner4.col) {
          // Same column
          houseType = `column ${corner3.col + 1}`
          for (let r = 0; r < size; r++) {
            if (r !== corner3.row && r !== corner4.row) {
              sharedHouseCells.push({ row: r, col: corner3.col })
            }
          }
        }

        if (sharedHouseCells.length === 0) continue

        // Find eliminations
        const eliminations: Elimination[] = []
        const highlights: CellHighlight[] = [
          { row: c1.cell.row, col: c1.cell.col, type: 'secondary', candidates: [A, B] },
          { row: c2.cell.row, col: c2.cell.col, type: 'secondary', candidates: [A, B] },
          { row: corner3.row, col: corner3.col, type: 'primary', candidates: [X] },
          { row: corner4.row, col: corner4.col, type: 'primary', candidates: [X] }
        ]

        for (const cell of sharedHouseCells) {
          const cellCands = candidates[cell.row]?.[cell.col] ?? []
          if (cellCands.includes(X)) {
            eliminations.push({ row: cell.row, col: cell.col, candidate: X })
            highlights.push({
              row: cell.row,
              col: cell.col,
              type: 'elimination',
              candidates: [X]
            })
          }
        }

        if (eliminations.length > 0) {
          return {
            strategyName: 'Unique Rectangle Type 2',
            difficulty: 4.5,
            description: `Unique Rectangle on {${A},${B}}. The extra candidate ${X} in ${formatCell(corner3.row, corner3.col)} and ${formatCell(corner4.row, corner4.col)} must contain the solution. ${X} can be eliminated from the rest of ${houseType}.`,
            eliminations,
            highlights
          }
        }
      }
    }
  }

  return null
}
