import type { Cell } from './types'

// Get all cells in a row
export function getRowCells(row: number, size: number): Cell[] {
  const cells: Cell[] = []
  for (let col = 0; col < size; col++) {
    cells.push({ row, col })
  }
  return cells
}

// Get all cells in a column
export function getColCells(col: number, size: number): Cell[] {
  const cells: Cell[] = []
  for (let row = 0; row < size; row++) {
    cells.push({ row, col })
  }
  return cells
}

// Get all cells in a box given the top-left corner
export function getBoxCells(
  boxStartRow: number,
  boxStartCol: number,
  boxHeight: number,
  boxWidth: number
): Cell[] {
  const cells: Cell[] = []
  for (let r = 0; r < boxHeight; r++) {
    for (let c = 0; c < boxWidth; c++) {
      cells.push({ row: boxStartRow + r, col: boxStartCol + c })
    }
  }
  return cells
}

// Get the top-left corner of the box containing a cell
export function getBoxStart(
  row: number,
  col: number,
  boxHeight: number,
  boxWidth: number
): { boxStartRow: number; boxStartCol: number } {
  return {
    boxStartRow: Math.floor(row / boxHeight) * boxHeight,
    boxStartCol: Math.floor(col / boxWidth) * boxWidth
  }
}

// Get all cells that "see" a given cell (same row, column, or box)
export function getPeers(
  row: number,
  col: number,
  size: number,
  boxHeight: number,
  boxWidth: number
): Cell[] {
  const peers: Cell[] = []
  const seen = new Set<string>()

  const addCell = (r: number, c: number) => {
    const key = `${r},${c}`
    if (!seen.has(key) && !(r === row && c === col)) {
      seen.add(key)
      peers.push({ row: r, col: c })
    }
  }

  // Row peers
  for (let c = 0; c < size; c++) {
    addCell(row, c)
  }

  // Column peers
  for (let r = 0; r < size; r++) {
    addCell(r, col)
  }

  // Box peers
  const { boxStartRow, boxStartCol } = getBoxStart(row, col, boxHeight, boxWidth)
  for (let r = 0; r < boxHeight; r++) {
    for (let c = 0; c < boxWidth; c++) {
      addCell(boxStartRow + r, boxStartCol + c)
    }
  }

  return peers
}

// Check if two cells see each other (share row, column, or box)
export function cellsSeEachOther(
  cell1: Cell,
  cell2: Cell,
  boxHeight: number,
  boxWidth: number
): boolean {
  // Same cell
  if (cell1.row === cell2.row && cell1.col === cell2.col) {
    return false
  }

  // Same row
  if (cell1.row === cell2.row) return true

  // Same column
  if (cell1.col === cell2.col) return true

  // Same box
  const box1 = getBoxStart(cell1.row, cell1.col, boxHeight, boxWidth)
  const box2 = getBoxStart(cell2.row, cell2.col, boxHeight, boxWidth)
  if (box1.boxStartRow === box2.boxStartRow && box1.boxStartCol === box2.boxStartCol) {
    return true
  }

  return false
}

// Get cells that see all given cells (intersection of peer sets)
export function getCellsSeeingAll(
  cells: Cell[],
  size: number,
  boxHeight: number,
  boxWidth: number
): Cell[] {
  if (cells.length === 0) return []

  // Start with all cells
  const result: Cell[] = []

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const candidate: Cell = { row: r, col: c }
      // Check if this cell sees all the given cells
      let seesAll = true
      for (const cell of cells) {
        if (!cellsSeEachOther(candidate, cell, boxHeight, boxWidth)) {
          seesAll = false
          break
        }
      }
      if (seesAll) {
        result.push(candidate)
      }
    }
  }

  return result
}

// Get all houses (rows, columns, boxes) as arrays of cells
export function getAllHouses(
  size: number,
  boxHeight: number,
  boxWidth: number
): { type: 'row' | 'col' | 'box'; index: number; cells: Cell[] }[] {
  const houses: { type: 'row' | 'col' | 'box'; index: number; cells: Cell[] }[] = []

  // Rows
  for (let r = 0; r < size; r++) {
    houses.push({ type: 'row', index: r, cells: getRowCells(r, size) })
  }

  // Columns
  for (let c = 0; c < size; c++) {
    houses.push({ type: 'col', index: c, cells: getColCells(c, size) })
  }

  // Boxes
  let boxIndex = 0
  for (let br = 0; br < size; br += boxHeight) {
    for (let bc = 0; bc < size; bc += boxWidth) {
      houses.push({ type: 'box', index: boxIndex, cells: getBoxCells(br, bc, boxHeight, boxWidth) })
      boxIndex++
    }
  }

  return houses
}

// Get the box index (0-8 for 9x9) for a cell
export function getBoxIndex(
  row: number,
  col: number,
  boxHeight: number,
  boxWidth: number
): number {
  const boxRow = Math.floor(row / boxHeight)
  const boxCol = Math.floor(col / boxWidth)
  const boxesPerRow = 9 / boxWidth // Assuming standard sizing
  return boxRow * boxesPerRow + boxCol
}

// Format cell reference for display (e.g., "R1C2")
export function formatCell(row: number, col: number): string {
  return `R${row + 1}C${col + 1}`
}

// Format multiple cells for display
export function formatCells(cells: Cell[]): string {
  return cells.map(c => formatCell(c.row, c.col)).join(', ')
}

// Get candidates as a sorted string for display (e.g., "{1,2,3}")
export function formatCandidates(candidates: number[]): string {
  return `{${[...candidates].sort((a, b) => a - b).join(',')}}`
}
