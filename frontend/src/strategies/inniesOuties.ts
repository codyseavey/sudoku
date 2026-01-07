import type { Strategy, SolveContext, HintResult, Elimination, CellHighlight, Cell, Cage } from './types'
import { formatCell } from './utils'

/**
 * Innies and Outies (Rule of 45):
 * In Killer Sudoku, each row, column, and box must sum to 45 (1+2+...+9).
 *
 * Innie: A cell from a cage that extends outside a house but has one cell inside.
 *        Innie value = 45 - sum of cages fully inside the house
 *
 * Outie: A cell from a cage mostly inside a house but with one cell outside.
 *        Outie value = sum of cages - 45
 *
 * When we can determine exact values, we can place digits or eliminate candidates.
 */
export const inniesOuties: Strategy = {
  name: 'Innies/Outies',
  difficulty: 2.0,

  find(context: SolveContext): HintResult | null {
    const { board, candidates, size, cages, boxHeight, boxWidth } = context

    // Only applies to Killer Sudoku
    if (!cages || cages.length === 0) {
      return null
    }

    // Try each row
    for (let row = 0; row < size; row++) {
      const result = analyzeHouse('row', row, board, candidates, cages, size, boxHeight, boxWidth)
      if (result) return result
    }

    // Try each column
    for (let col = 0; col < size; col++) {
      const result = analyzeHouse('col', col, board, candidates, cages, size, boxHeight, boxWidth)
      if (result) return result
    }

    // Try each box
    for (let boxIdx = 0; boxIdx < size; boxIdx++) {
      const result = analyzeHouse('box', boxIdx, board, candidates, cages, size, boxHeight, boxWidth)
      if (result) return result
    }

    return null
  }
}

function analyzeHouse(
  houseType: 'row' | 'col' | 'box',
  houseIndex: number,
  board: (number | null)[][],
  candidates: number[][][],
  cages: Cage[],
  size: number,
  boxHeight: number,
  boxWidth: number
): HintResult | null {
  // Get cells in this house
  const houseCells = getHouseCells(houseType, houseIndex, size, boxHeight, boxWidth)
  const houseCellSet = new Set(houseCells.map(c => `${c.row},${c.col}`))

  // Categorize cages relative to this house
  let sumFullyInside = 0
  const cagesFullyInside: Cage[] = []
  const innies: { cage: Cage; cellInside: Cell }[] = []
  const outies: { cage: Cage; cellOutside: Cell }[] = []

  for (const cage of cages) {
    const cellsInHouse = cage.cells.filter(c => houseCellSet.has(`${c.row},${c.col}`))
    const cellsOutside = cage.cells.filter(c => !houseCellSet.has(`${c.row},${c.col}`))

    if (cellsOutside.length === 0) {
      // Cage fully inside house
      cagesFullyInside.push(cage)
      sumFullyInside += cage.sum
    } else if (cellsInHouse.length === 1 && cellsOutside.length === cage.cells.length - 1) {
      // One cell inside (innie)
      innies.push({ cage, cellInside: cellsInHouse[0]! })
    } else if (cellsOutside.length === 1 && cellsInHouse.length === cage.cells.length - 1) {
      // One cell outside (outie)
      outies.push({ cage, cellOutside: cellsOutside[0]! })
    }
  }

  // Calculate known sum from solved cells and fully-inside cages
  let knownSum = 0
  let unsolvedCellsNotInCages: Cell[] = []

  for (const cell of houseCells) {
    const value = board[cell.row]?.[cell.col]
    if (value !== null && value !== undefined) {
      knownSum += value
    } else if (value === null) {
      // Check if this cell is in a fully-inside cage
      const inFullCage = cagesFullyInside.some(cage =>
        cage.cells.some(cc => cc.row === cell.row && cc.col === cell.col)
      )
      if (!inFullCage) {
        // This is an innie or part of crossing cage
        const innieEntry = innies.find(i =>
          i.cellInside.row === cell.row && i.cellInside.col === cell.col
        )
        if (!innieEntry) {
          unsolvedCellsNotInCages.push(cell)
        }
      }
    }
  }

  // Simple innie calculation: if we have one unknown cell that's an innie
  if (innies.length === 1 && unsolvedCellsNotInCages.length === 0) {
    const innie = innies[0]!
    const cellValue = board[innie.cellInside.row]?.[innie.cellInside.col]

    if (cellValue === null) {
      // Calculate what the innie must be
      // Sum of house = 45
      // Sum of fully inside cages + innie contribution = 45
      // But we need to account for the cage sum that includes the innie

      // Actually, simpler approach:
      // If all cells in house are accounted for by cages, and one cage has exactly one cell
      // crossing the boundary...

      // The innie cell's value can be determined from the cage sum minus other cells

      // Get sum of solved cells in the innie's cage (outside the house)
      let cageSolvedSum = 0
      let cageUnsolvedOutside: Cell[] = []

      for (const cageCell of innie.cage.cells) {
        if (cageCell.row === innie.cellInside.row && cageCell.col === innie.cellInside.col) {
          continue // Skip the innie cell itself
        }
        const val = board[cageCell.row]?.[cageCell.col]
        if (val !== null && val !== undefined) {
          cageSolvedSum += val
        } else if (val === null) {
          cageUnsolvedOutside.push(cageCell)
        }
      }

      // If all other cells in the cage are solved, we know the innie value
      if (cageUnsolvedOutside.length === 0) {
        const innieValue = innie.cage.sum - cageSolvedSum

        if (innieValue >= 1 && innieValue <= 9) {
          const cellCands = candidates[innie.cellInside.row]?.[innie.cellInside.col] ?? []

          if (cellCands.includes(innieValue) && cellCands.length > 1) {
            // Can eliminate other candidates
            const eliminations: Elimination[] = cellCands
              .filter(c => c !== innieValue)
              .map(c => ({
                row: innie.cellInside.row,
                col: innie.cellInside.col,
                candidate: c
              }))

            const highlights: CellHighlight[] = [
              {
                row: innie.cellInside.row,
                col: innie.cellInside.col,
                type: 'primary',
                candidates: [innieValue]
              }
            ]

            // Highlight the cage cells
            for (const cageCell of innie.cage.cells) {
              if (cageCell.row !== innie.cellInside.row || cageCell.col !== innie.cellInside.col) {
                highlights.push({
                  row: cageCell.row,
                  col: cageCell.col,
                  type: 'secondary',
                  candidates: []
                })
              }
            }

            if (eliminations.length > 0) {
              const houseTypeName = houseType === 'row' ? `row ${houseIndex + 1}` :
                                    houseType === 'col' ? `column ${houseIndex + 1}` :
                                    `box ${houseIndex + 1}`

              return {
                strategyName: 'Innies/Outies',
                difficulty: 2.0,
                description: `Innie in ${houseTypeName}: Cell ${formatCell(innie.cellInside.row, innie.cellInside.col)} is the only part of its cage (sum ${innie.cage.sum}) inside the house. Based on the cage sum, it must be ${innieValue}.`,
                eliminations,
                highlights
              }
            }
          }
        }
      }
    }
  }

  // Simple outie calculation
  if (outies.length === 1) {
    const outie = outies[0]!
    const cellValue = board[outie.cellOutside.row]?.[outie.cellOutside.col]

    if (cellValue === null) {
      // Get sum of solved cells in the outie's cage (inside the house)
      let cageSolvedSum = 0
      let cageUnsolvedInside: Cell[] = []

      for (const cageCell of outie.cage.cells) {
        if (cageCell.row === outie.cellOutside.row && cageCell.col === outie.cellOutside.col) {
          continue // Skip the outie cell itself
        }
        const val = board[cageCell.row]?.[cageCell.col]
        if (val !== null && val !== undefined) {
          cageSolvedSum += val
        } else if (val === null) {
          cageUnsolvedInside.push(cageCell)
        }
      }

      // If all cells inside the house are solved, we know the outie value
      if (cageUnsolvedInside.length === 0) {
        const outieValue = outie.cage.sum - cageSolvedSum

        if (outieValue >= 1 && outieValue <= 9) {
          const cellCands = candidates[outie.cellOutside.row]?.[outie.cellOutside.col] ?? []

          if (cellCands.includes(outieValue) && cellCands.length > 1) {
            const eliminations: Elimination[] = cellCands
              .filter(c => c !== outieValue)
              .map(c => ({
                row: outie.cellOutside.row,
                col: outie.cellOutside.col,
                candidate: c
              }))

            const highlights: CellHighlight[] = [
              {
                row: outie.cellOutside.row,
                col: outie.cellOutside.col,
                type: 'primary',
                candidates: [outieValue]
              }
            ]

            for (const cageCell of outie.cage.cells) {
              if (cageCell.row !== outie.cellOutside.row || cageCell.col !== outie.cellOutside.col) {
                highlights.push({
                  row: cageCell.row,
                  col: cageCell.col,
                  type: 'secondary',
                  candidates: []
                })
              }
            }

            if (eliminations.length > 0) {
              const houseTypeName = houseType === 'row' ? `row ${houseIndex + 1}` :
                                    houseType === 'col' ? `column ${houseIndex + 1}` :
                                    `box ${houseIndex + 1}`

              return {
                strategyName: 'Innies/Outies',
                difficulty: 2.0,
                description: `Outie from ${houseTypeName}: Cell ${formatCell(outie.cellOutside.row, outie.cellOutside.col)} is the only part of its cage (sum ${outie.cage.sum}) outside the house. Based on the cage sum, it must be ${outieValue}.`,
                eliminations,
                highlights
              }
            }
          }
        }
      }
    }
  }

  return null
}

function getHouseCells(
  houseType: 'row' | 'col' | 'box',
  houseIndex: number,
  size: number,
  boxHeight: number,
  boxWidth: number
): Cell[] {
  const cells: Cell[] = []

  if (houseType === 'row') {
    for (let c = 0; c < size; c++) {
      cells.push({ row: houseIndex, col: c })
    }
  } else if (houseType === 'col') {
    for (let r = 0; r < size; r++) {
      cells.push({ row: r, col: houseIndex })
    }
  } else {
    // Box
    const boxesPerRow = size / boxWidth
    const boxRow = Math.floor(houseIndex / boxesPerRow)
    const boxCol = houseIndex % boxesPerRow
    const startRow = boxRow * boxHeight
    const startCol = boxCol * boxWidth

    for (let r = startRow; r < startRow + boxHeight; r++) {
      for (let c = startCol; c < startCol + boxWidth; c++) {
        cells.push({ row: r, col: c })
      }
    }
  }

  return cells
}
