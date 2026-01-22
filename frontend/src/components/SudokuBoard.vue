<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted } from 'vue'
import { findHint } from '../strategies'
import type { HintResult, SolveContext } from '../strategies/types'

const props = defineProps<{
    initialDifficulty: string,
    size: number,
    initialPuzzle?: Record<string, any>,
    isCustomMode?: boolean,
    gameType?: string,
    initialBoardUpload?: number[][],
    initialCages?: { sum: number, cells: { row: number, col: number }[] }[]
}>()

const emit = defineEmits(['back-to-menu', 'puzzle-completed'])

// Use null for empty cells instead of 0
const board = ref<(number | null)[][]>([])
const isFixed = ref<boolean[][]>([])
const solution = ref<number[][]>([])
const initialBoardState = ref<number[][]>([]) // To store the clean initial state for sharing
const cages = ref<{ sum: number, cells: { row: number, col: number }[] }[]>([])

const candidates = ref<number[][][]>([])
// Track candidates that have been explicitly removed
const eliminatedCandidates = ref<Set<number>[][]>([])

const loading = ref(false)
const message = ref('')
const isNoteMode = ref(false)
const difficulty = ref(props.initialDifficulty)
const isDefiningCustom = ref(false) // State for entering custom puzzle
const timer = ref(0)
const isPaused = ref(false)
let timerInterval: number | undefined
const isMobile = ref(false)
const selectedCell = ref<{r: number, c: number} | null>(null)
const isGameOver = ref(false)

// Hint system state
const hintResult = ref<HintResult | null>(null)
const showHintHighlights = ref(false)
const candidatesPopulated = ref(false)

// Custom Cage Creation Logic
const isCageSelectionMode = ref(false)
const currentCageSelection = ref<Set<string>>(new Set()) // Store as "r,c" strings

const STORAGE_KEY = `sudoku_game_state_${props.size}`

// Timer Logic
const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

const startTimer = () => {
    if (timerInterval) return
    isPaused.value = false
    timerInterval = window.setInterval(() => {
        timer.value++
    }, 1000)
}

const pauseTimer = () => {
    if (timerInterval) {
        clearInterval(timerInterval)
        timerInterval = undefined
    }
    isPaused.value = true
}

const resumeTimer = () => {
    startTimer()
}

const stopTimer = () => {
    if (timerInterval) {
        clearInterval(timerInterval)
        timerInterval = undefined
    }
}

// Initialize arrays based on size
const initArrays = () => {
    const s = props.size
    board.value = Array.from({ length: s }, () => Array(s).fill(null))
    isFixed.value = Array.from({ length: s }, () => Array(s).fill(false))
    solution.value = []
    candidates.value = Array.from({ length: s }, () => Array.from({ length: s }, () => [] as number[]))
    eliminatedCandidates.value = Array.from({ length: s }, () => Array.from({ length: s }, () => new Set<number>()))
    // Reset hint state
    candidatesPopulated.value = false
    hintResult.value = null
    showHintHighlights.value = false
}

// History Stack for Undo
interface GameState {
    board: (number | null)[][];
    candidates: number[][][];
    eliminatedCandidates: Set<number>[][];
}
const history = ref<GameState[]>([])

// Deep copy helper
const cloneState = (): GameState => {
    return {
        board: board.value.map(row => [...row]),
        candidates: candidates.value.map(row => row.map(cell => [...cell])),
        eliminatedCandidates: eliminatedCandidates.value.map(row => row.map(set => new Set(set)))
    }
}

const saveState = () => {
    history.value.push(cloneState())
    // Optional: Limit history size
    if (history.value.length > 50) history.value.shift()
}

const undo = () => {
    if (isGameOver.value || history.value.length === 0) return
    const prevState = history.value.pop()
    if (prevState) {
        board.value = prevState.board
        candidates.value = prevState.candidates
        eliminatedCandidates.value = prevState.eliminatedCandidates
        // Clear hints when undoing
        hintResult.value = null
        showHintHighlights.value = false
        message.value = 'Undo successful.'
        saveGame() // Sync with local storage
    }
}

const saveGame = () => {
    const serializeHistory = (hist: GameState[]) => {
        return hist.map(state => ({
            board: state.board,
            candidates: state.candidates,
            eliminatedCandidates: state.eliminatedCandidates.map(row => 
                row.map(set => Array.from(set))
            )
        }))
    }

    const gameState = {
        board: board.value,
        isFixed: isFixed.value,
        solution: solution.value,
        initialBoardState: initialBoardState.value, // Save for sharing later
        cages: cages.value,
        candidates: candidates.value,
        // Convert Sets to Arrays for JSON serialization
        eliminatedCandidates: eliminatedCandidates.value.map(row => 
            row.map(set => Array.from(set))
        ),
        difficulty: difficulty.value,
        history: serializeHistory(history.value),
        size: props.size,
        isCustomMode: props.isCustomMode,
        isDefiningCustom: isDefiningCustom.value,
        timer: timer.value,
        gameType: props.gameType
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(gameState))
}

const loadUploadedBoard = (data: number[][]) => {
    // Setup from upload (similar to defining custom, but prefilled)
    board.value = data.map(row => row.map(val => val === 0 ? null : val))
    isFixed.value = Array.from({ length: props.size }, () => Array(props.size).fill(false)) // Editable
    solution.value = [] // Not solved yet
    initialBoardState.value = []
    isDefiningCustom.value = true

    if (props.initialCages) {
        cages.value = props.initialCages
    } else {
        cages.value = []
    }

    resetCandidates()
    history.value = []
    timer.value = 0
    stopTimer()
    saveGame()
    message.value = 'Puzzle uploaded. Please verify and edit if needed.'
}

const loadGame = (): boolean => {
    // If we have an imported puzzle, use it strictly
    if (props.initialPuzzle) {
        loadImportedPuzzle(props.initialPuzzle)
        return true
    }

    // If we have an uploaded board for custom mode, use it
    if (props.isCustomMode && props.initialBoardUpload) {
        loadUploadedBoard(props.initialBoardUpload)
        return true
    }

    // If newly entering custom mode (and not loading a save), we skip loading
    // But we need to check if there is a save first to allow refresh in custom mode.
    // If props.isCustomMode is true, we should check if the saved game is also custom.

    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
        try {
            const gameState = JSON.parse(saved)
            if (gameState.size !== props.size) return false // Mismatch size
            
            // Check if saved game mode matches current intention
            if (props.isCustomMode && !gameState.isCustomMode) return false // Wanted custom, found normal
            if (!props.isCustomMode && gameState.isCustomMode) return false // Wanted normal, found custom

            // Check game type mismatch
            // Handle legacy saves where gameType is undefined (treat as 'standard')
            const savedType = gameState.gameType || 'standard'
            const currentType = props.gameType || 'standard'
            if (savedType !== currentType) return false

            board.value = gameState.board
            isFixed.value = gameState.isFixed
            solution.value = gameState.solution
            initialBoardState.value = gameState.initialBoardState || []
            cages.value = gameState.cages || []
            candidates.value = gameState.candidates
            // Convert Arrays back to Sets
            eliminatedCandidates.value = gameState.eliminatedCandidates.map((row: number[][]) => 
                row.map((arr: number[]) => new Set(arr))
            )
            if (gameState.difficulty) {
                difficulty.value = gameState.difficulty
            }
            if (gameState.history) {
                 history.value = gameState.history.map((state: any) => ({
                    board: state.board,
                    candidates: state.candidates,
                    eliminatedCandidates: state.eliminatedCandidates.map((row: number[][]) => 
                        row.map((arr: number[]) => new Set(arr))
                    )
                 }))
            }
            if (gameState.isDefiningCustom !== undefined) {
                isDefiningCustom.value = gameState.isDefiningCustom
            }
            if (gameState.timer !== undefined) {
                timer.value = gameState.timer
            }
            return true
        } catch (e) {
            console.error('Failed to load game state', e)
            return false
        }
    }
    return false
}

const loadImportedPuzzle = (data: any) => {
    // Setup from import
    board.value = data.board.map((row: number[]) => row.map(val => val === 0 ? null : val))
    isFixed.value = data.board.map((row: number[]) => row.map(val => val !== 0))
    solution.value = data.solution
    initialBoardState.value = data.board // Keep original for re-sharing
    difficulty.value = data.difficulty
    
    // Clear history/storage for this "new" game
    resetCandidates()
    history.value = []
    timer.value = 0
    startTimer()
    saveGame()
    message.value = 'Puzzle imported successfully.'
}

const startNewGame = () => {
    stopTimer()
    emit('back-to-menu')
}

const validateAndStartCustom = async () => {
    loading.value = true
    message.value = 'Validating puzzle...'
    
    // Prepare board for API (0 for null)
    const boardData = board.value.map(row => row.map(cell => cell === null ? 0 : cell))

    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardData, size: props.size })
        })

        if (!response.ok) {
            const err = await response.text()
            message.value = `Invalid puzzle: ${err.replace(/\n/g, ' ')}`
            loading.value = false
            return
        }

        const data = await response.json()
        solution.value = data.solution
        
        // Lock currently entered cells
        isFixed.value = board.value.map(row => row.map(cell => cell !== null))
        initialBoardState.value = boardData
        
        // Start Game
        isDefiningCustom.value = false
        message.value = 'Puzzle valid! Game started.'
        timer.value = 0
        startTimer()
        saveGame()

    } catch (e) {
        console.error(e)
        message.value = 'Error validating puzzle.'
    } finally {
        loading.value = false
    }
}

const shareGame = () => {
    if (!initialBoardState.value || initialBoardState.value.length === 0) {
        // If sharing in custom definition mode, we construct state from current board
        if (isDefiningCustom.value) {
             // Note: Solution might not exist yet, so import will need to solve it or we block sharing until solved?
             // Ideally block until solved to ensure validity.
             message.value = 'Please solve/validate the puzzle before sharing.'
             return
        }
        message.value = 'Cannot share this puzzle.'
        return
    }
    
    const data = {
        board: initialBoardState.value,
        solution: solution.value,
        size: props.size,
        difficulty: difficulty.value
    }
    
    const json = JSON.stringify(data)
    const base64 = btoa(json)
    const url = `${window.location.origin}/?import=${base64}`
    
    navigator.clipboard.writeText(url).then(() => {
        message.value = 'Link copied to clipboard!'
        setTimeout(() => message.value = '', 3000)
    }).catch(err => {
        console.error('Failed to copy text: ', err)
        message.value = 'Failed to copy link.'
    })
}

const fetchPuzzle = async () => {
  if (props.isCustomMode) {
      isDefiningCustom.value = true
      return
  }

  loading.value = true
  message.value = ''
  try {
    let url = `/api/puzzle?difficulty=${difficulty.value}&size=${props.size}`
    if (props.gameType) {
        url += `&gameType=${props.gameType}`
    }
    const response = await fetch(url)
    const data = await response.json()
    // Convert 0s to nulls for the UI
    board.value = data.board.map((row: number[]) => row.map(val => val === 0 ? null : val))
    // Mark non-empty cells as fixed
    isFixed.value = data.board.map((row: number[]) => row.map(val => val !== 0))
    
    solution.value = data.solution
    initialBoardState.value = data.board // Store raw
    cages.value = data.cages || []
    
    // Reset candidates and elimination history
    resetCandidates()
    history.value = [] // Reset history
    timer.value = 0
    startTimer()
    saveGame() // Save initial state
  } catch (error) {
    console.error('Error fetching puzzle:', error)
    message.value = 'Failed to load puzzle.'
  } finally {
    loading.value = false
  }
}

const resetCandidates = () => {
    const s = props.size
    candidates.value = Array.from({ length: s }, () => Array.from({ length: s }, () => [] as number[]))
    eliminatedCandidates.value = Array.from({ length: s }, () => Array.from({ length: s }, () => new Set<number>()))
}

// Watch for changes to save state
watch([board, candidates, eliminatedCandidates, difficulty], () => {
    saveGame()
}, { deep: true })

const checkSolution = () => {
    // 1. Basic Completeness Check
    if (!isBoardFull()) {
        message.value = 'Incomplete! Fill all cells.'
        return
    }

    // 2. Validate Rules (Rows, Cols, Boxes)
    // Rows
    for (let r = 0; r < props.size; r++) {
        const rowVals = new Set<number>()
        for (let c = 0; c < props.size; c++) {
            const val = board.value[r]![c]!
            if (rowVals.has(val)) {
                message.value = `Duplicate number ${val} in Row ${r + 1}`
                return
            }
            rowVals.add(val)
        }
    }

    // Cols
    for (let c = 0; c < props.size; c++) {
        const colVals = new Set<number>()
        for (let r = 0; r < props.size; r++) {
            const val = board.value[r]![c]!
            if (colVals.has(val)) {
                message.value = `Duplicate number ${val} in Column ${c + 1}`
                return
            }
            colVals.add(val)
        }
    }

    // Boxes
    const boxH = props.size === 6 ? 2 : 3
    const boxW = 3
    for (let br = 0; br < props.size; br += boxH) {
        for (let bc = 0; bc < props.size; bc += boxW) {
             const boxVals = new Set<number>()
             for (let i = 0; i < boxH; i++) {
                 for (let j = 0; j < boxW; j++) {
                     const val = board.value[br + i]![bc + j]!
                     if (boxVals.has(val)) {
                         message.value = `Duplicate number ${val} in Box`
                         return
                     }
                     boxVals.add(val)
                 }
             }
        }
    }

    // 3. Killer Rules (if applicable)
    if (props.gameType === 'killer') {
        for (const cage of cages.value) {
            let currentSum = 0
            const cageVals = new Set<number>()
            for (const cell of cage.cells) {
                const val = board.value[cell.row]![cell.col]!
                if (cageVals.has(val)) {
                    message.value = `Duplicate number ${val} in a cage.`
                    return
                }
                cageVals.add(val)
                currentSum += val
            }
            if (currentSum !== cage.sum) {
                message.value = `Cage sum mismatch. Expected ${cage.sum}, got ${currentSum}.`
                return
            }
        }
    }

    // If we passed all checks
    message.value = 'Correct! Well done.'
    stopTimer()

    // Send completion to backend
    fetch('/api/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            difficulty: props.initialDifficulty,
            size: props.size,
            gameType: props.gameType || 'standard'
        })
    }).catch(e => console.error('Failed to report completion', e))

    // Notify parent
    emit('puzzle-completed', { board: board.value })
}

const isBoardFull = () => {
    return board.value.every(row => row.every(cell => cell !== null))
}

const showSolution = () => {
    if (solution.value.length === 0) return
    board.value = solution.value.map(row => [...row])
    resetCandidates()
    message.value = 'Solution revealed.'
    stopTimer()
    isGameOver.value = true
    history.value = [] // Clear undo history
    // End the game by clearing the saved state
    localStorage.removeItem(STORAGE_KEY)
}

const removeCandidates = (r: number, c: number, val: number) => {
    const removeFromCell = (row: number, col: number, v: number) => {
        const idx = candidates.value[row]![col]!.indexOf(v)
        if (idx > -1) {
            candidates.value[row]![col]!.splice(idx, 1)
            // Mark as eliminated so it doesn't come back with auto-candidates
            eliminatedCandidates.value[row]![col]!.add(v)
        }
    }

    const s = props.size
    const boxH = s === 6 ? 2 : 3
    const boxW = 3

    // Row
    for (let j = 0; j < s; j++) removeFromCell(r, j, val)
    // Col
    for (let i = 0; i < s; i++) removeFromCell(i, c, val)
    // Box
    const startRow = Math.floor(r / boxH) * boxH
    const startCol = Math.floor(c / boxW) * boxW
    for (let i = 0; i < boxH; i++) {
        for (let j = 0; j < boxW; j++) {
            removeFromCell(startRow + i, startCol + j, val)
        }
    }
}

// Interaction Handler
const handleKeydown = (e: KeyboardEvent, r: number, c: number) => {
    const key = e.key
    
    // Ctrl+Z for Undo
    if (key === 'z' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        undo()
        return
    }

    // Allow navigation and deletion normally
    if (['Backspace', 'Delete', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(key)) {
        if (['Backspace', 'Delete'].includes(key)) {
            if (!isGameOver.value && !isFixed.value[r]![c] && board.value[r]![c] !== null) {
                 saveState() // Save before clearing
                 board.value[r]![c] = null
            }
        }
        return
    }
}

const onFocus = (r: number, c: number) => {
    if (isCageSelectionMode.value) return // Don't focus if selecting cages
    selectedCell.value = { r, c }
}

const toggleCageSelectionMode = () => {
    isCageSelectionMode.value = !isCageSelectionMode.value
    currentCageSelection.value.clear()
    selectedCell.value = null
}

const onCellClickInCageMode = (r: number, c: number) => {
    if (!isCageSelectionMode.value) return
    const key = `${r},${c}`
    if (currentCageSelection.value.has(key)) {
        currentCageSelection.value.delete(key)
    } else {
        currentCageSelection.value.add(key)
    }
}

const saveCage = () => {
    if (currentCageSelection.value.size === 0) return

    const sumStr = prompt('Enter sum for this cage:')
    if (!sumStr) return

    const sum = parseInt(sumStr)
    if (isNaN(sum)) {
        alert('Invalid sum')
        return
    }

    const newCageCells: { row: number, col: number }[] = []
    currentCageSelection.value.forEach(key => {
        const parts = key.split(',').map(Number)
        if (parts.length >= 2) {
            newCageCells.push({ row: parts[0]!, col: parts[1]! })
        }
    })

    cages.value.push({ sum, cells: newCageCells })
    currentCageSelection.value.clear()
    isCageSelectionMode.value = false
}

const cancelCageSelection = () => {
    currentCageSelection.value.clear()
    isCageSelectionMode.value = false
}

const deleteCage = () => {
    if (!selectedCell.value) return
    const { r, c } = selectedCell.value
    const idx = cages.value.findIndex(cg => cg.cells.some(cell => cell.row === r && cell.col === c))
    if (idx > -1 && cages.value[idx]) {
        if (confirm(`Delete cage with sum ${cages.value[idx].sum}?`)) {
            cages.value.splice(idx, 1)
        }
    }
}

const setCellValue = (r: number, c: number, num: number) => {
    // In custom definition mode, we don't check isFixed (everything is editable initially)
    // Once game starts, we check isFixed.
    if (isGameOver.value) return
    if (!isDefiningCustom.value && isFixed.value[r]?.[c]) return

    saveState()

    // Logic for Custom Definition Mode
    if (isDefiningCustom.value) {
        if (board.value[r]) board.value[r]![c] = num
        return
    }

    // Logic for Normal Play Mode
    if (isNoteMode.value) {
        const currentCandidates = candidates.value[r]?.[c]
        if (!currentCandidates) return
        const idx = currentCandidates.indexOf(num)
        if (idx > -1) {
            currentCandidates.splice(idx, 1)
            eliminatedCandidates.value[r]![c]!.add(num)
        } else {
            currentCandidates.push(num)
            currentCandidates.sort()
            eliminatedCandidates.value[r]![c]!.delete(num)
        }
    } else {
        if (board.value[r]) board.value[r]![c] = num
        if (candidates.value[r]) candidates.value[r]![c] = []

        // Auto Eliminate Peer Candidates (Always ON)
        if (num !== undefined && num === solution.value[r]![c]) {
            removeCandidates(r, c, num)
        }

        // Auto Check if full
        if (isBoardFull()) {
            checkSolution()
        }
    }
}

const onPadInput = (num: number) => {
    if (!selectedCell.value) return
    const { r, c } = selectedCell.value
    setCellValue(r, c, num)
}

const onPadClear = () => {
    if (!selectedCell.value || isGameOver.value) return
    const { r, c } = selectedCell.value
    if (!isFixed.value[r]![c] && board.value[r]![c] !== null) {
        saveState() // Save before clearing
        board.value[r]![c] = null
    }
}

const handleInput = (e: Event, r: number, c: number) => {
    const input = e.target as HTMLInputElement
    const val = input.value
    
    // Reset input value immediately
    input.value = board.value[r]?.[c]?.toString() || ''

    if (!val) return

    const char = val.slice(-1)
    // Check if 1-N (size)
    const max = props.size
    const regex = new RegExp(`^[1-${max}]$`)

    if (regex.test(char)) {
        const num = parseInt(char)
        setCellValue(r, c, num)
    }
}

// Auto Candidate Logic
const generateCandidates = () => {
    saveState() // Save before generating
    for (let r = 0; r < props.size; r++) {
        for (let c = 0; c < props.size; c++) {
            if (board.value[r]![c] === null) {
                const valid = getValidNumbers(r, c)
                // Filter out candidates that have been explicitly eliminated
                candidates.value[r]![c] = valid.filter(n => !eliminatedCandidates.value[r]![c]!.has(n))
            } else {
                candidates.value[r]![c] = []
            }
        }
    }
    candidatesPopulated.value = true
    // Clear any previous hint when refreshing candidates
    hintResult.value = null
    showHintHighlights.value = false
    message.value = 'Candidates generated (respecting eliminations.)'
}

// Hint System Functions
const getHint = () => {
    if (!candidatesPopulated.value) {
        message.value = 'Please populate candidates first.'
        return
    }

    const context: SolveContext = {
        board: board.value,
        candidates: candidates.value,
        size: props.size,
        boxHeight: props.size === 6 ? 2 : 3,
        boxWidth: 3,
        gameType: (props.gameType as 'standard' | 'killer') || 'standard',
        cages: cages.value.length > 0 ? cages.value : undefined,
    }

    const hint = findHint(context)

    if (hint) {
        hintResult.value = hint
        showHintHighlights.value = true
        message.value = `${hint.strategyName}: ${hint.description}`
    } else {
        hintResult.value = null
        showHintHighlights.value = false
        message.value = 'No hint available. The puzzle may be solved or require more advanced techniques.'
    }
}

const applyHint = () => {
    if (!hintResult.value || hintResult.value.eliminations.length === 0) {
        message.value = 'No eliminations to apply.'
        return
    }

    saveState()

    for (const elim of hintResult.value.eliminations) {
        const idx = candidates.value[elim.row]?.[elim.col]?.indexOf(elim.candidate)
        if (idx !== undefined && idx > -1) {
            candidates.value[elim.row]![elim.col]!.splice(idx, 1)
            eliminatedCandidates.value[elim.row]![elim.col]!.add(elim.candidate)
        }
    }

    message.value = `Applied ${hintResult.value.strategyName}: ${hintResult.value.eliminations.length} candidate(s) eliminated.`
    hintResult.value = null
    showHintHighlights.value = false
}

const dismissHint = () => {
    hintResult.value = null
    showHintHighlights.value = false
    message.value = ''
}

// Helper function to get highlight class for a cell
const getHintHighlightClass = (row: number, col: number): Record<string, boolean> => {
    if (!showHintHighlights.value || !hintResult.value) return {}

    const highlight = hintResult.value.highlights.find(h => h.row === row && h.col === col)
    if (!highlight) return {}

    return {
        'hint-primary': highlight.type === 'primary',
        'hint-secondary': highlight.type === 'secondary',
        'hint-elimination': highlight.type === 'elimination',
    }
}

// Helper function to check if a specific candidate should be highlighted
const getCandidateHighlightClass = (row: number, col: number, num: number): Record<string, boolean> => {
    if (!showHintHighlights.value || !hintResult.value) return {}

    const highlight = hintResult.value.highlights.find(h =>
        h.row === row && h.col === col && h.candidates?.includes(num)
    )
    if (!highlight) return {}

    return {
        'candidate-highlight-primary': highlight.type === 'primary',
        'candidate-highlight-elimination': highlight.type === 'elimination',
    }
}

const getValidNumbers = (row: number, col: number): number[] => {
    const used = new Set<number>()
    const s = props.size
    
    // Check Row
    for (let c = 0; c < s; c++) {
        const val = board.value[row]?.[c]
        if (typeof val === 'number') used.add(val)
    }

    // Check Col
    for (let r = 0; r < s; r++) {
        const val = board.value[r]?.[col]
        if (typeof val === 'number') used.add(val)
    }

    // Check Box
    const boxH = s === 6 ? 2 : 3
    const boxW = 3
    const startRow = Math.floor(row / boxH) * boxH
    const startCol = Math.floor(col / boxW) * boxW
    for (let r = 0; r < boxH; r++) {
        for (let c = 0; c < boxW; c++) {
            const val = board.value[startRow + r]?.[startCol + c]
            if (typeof val === 'number') used.add(val)
        }
    }

    const valid: number[] = []
    for (let n = 1; n <= s; n++) {
        if (!used.has(n)) valid.push(n)
    }
    return valid
}

const getCageStyle = (r: number, c: number) => {
    if (!props.gameType || props.gameType !== 'killer') return {}

    // Find the cage this cell belongs to
    const cage = cages.value.find(cg => cg.cells.some(cell => cell.row === r && cell.col === c))
    if (!cage) return {}

    const isSameCage = (nr: number, nc: number) => {
         return cage.cells.some(cell => cell.row === nr && cell.col === nc)
    }

    return {
        'cage-border-top': !isSameCage(r - 1, c),
        'cage-border-bottom': !isSameCage(r + 1, c),
        'cage-border-left': !isSameCage(r, c - 1),
        'cage-border-right': !isSameCage(r, c + 1)
    }
}

const getCageSum = (r: number, c: number) => {
    if (!props.gameType || props.gameType !== 'killer') return null
    const cage = cages.value.find(cg => cg.cells.some(cell => cell.row === r && cell.col === c))
    if (!cage) return null

    // Display sum only in the top-left-most cell of the cage
    // Sort cells by row then col to find "first" cell
    const sorted = [...cage.cells].sort((a, b) => {
        if (a.row !== b.row) return a.row - b.row
        return a.col - b.col
    })

    if (sorted.length > 0 && sorted[0] && sorted[0].row === r && sorted[0].col === c) {
        return cage.sum
    }
    return null
}

const updateMobileState = () => {
    isMobile.value = window.innerWidth < 768
}

onMounted(() => {
    updateMobileState()
    window.addEventListener('resize', updateMobileState)
    initArrays()
    if (loadGame()) {
        // Resume timer if game loaded and not defining custom
        if (!isDefiningCustom.value) {
            startTimer()
        }
    } else {
        fetchPuzzle()
    }
})

onUnmounted(() => {
    window.removeEventListener('resize', updateMobileState)
    stopTimer()
})
</script>

<template>
  <div class="board-container">
    <div v-if="loading" class="loading">Generating Puzzle...</div>
    
    <div v-else class="game-area">
      <div class="header-status" v-if="isDefiningCustom">
        <h3>Enter Your Puzzle</h3>
        <p>Fill in the initial numbers.</p>
      </div>

      <div class="timer-display" v-else>
        {{ formatTime(timer) }}
      </div>

      <div class="grid-outer">
        <div class="grid" :class="`size-${size}`">
          <div v-for="(row, rIndex) in board" :key="rIndex" class="row">
          <div
            v-for="(cell, cIndex) in row"
            :key="cIndex"
            class="cell"
            :class="[
              getCageStyle(rIndex, cIndex),
              { 'cage-selected': currentCageSelection.has(`${rIndex},${cIndex}`) },
              getHintHighlightClass(rIndex, cIndex)
            ]"
            @click="onCellClickInCageMode(rIndex, cIndex)"
          >
            <div class="cell-inner">
              <div v-if="getCageSum(rIndex, cIndex)" class="cage-sum">{{ getCageSum(rIndex, cIndex) }}</div>
              <!-- Main Value Input -->
              <input 
                type="text"
                inputmode="numeric"
                :value="cell"
                :readonly="isMobile || isCageSelectionMode"
                @focus="onFocus(rIndex, cIndex)"
                @keydown="handleKeydown($event, rIndex, cIndex)"
                @input="handleInput($event, rIndex, cIndex)"
                autocomplete="off"
                class="value-input"
                :class="{
                  'hidden': cell === null,
                  'fixed': isFixed[rIndex]![cIndex] || isDefiningCustom,
                  'pointer-events-none': isCageSelectionMode
                }"
              />
                  
              <!-- Incorrect Mark (Red X) -->
              <div v-if="!isDefiningCustom && !isFixed[rIndex]![cIndex] && cell !== null && cell !== solution[rIndex]![cIndex]" class="incorrect-mark">
                X
              </div>

              <!-- Candidates Overlay -->
              <div v-if="!isDefiningCustom && cell === null && candidates[rIndex]![cIndex]!.length > 0" class="candidates-grid" :class="[`size-${size}`, { 'killer-mode': gameType === 'killer' }]">
                <div
                  v-for="num in size"
                  :key="num"
                  class="candidate-cell"
                  :class="getCandidateHighlightClass(rIndex, cIndex, num)"
                >
                  {{ candidates[rIndex]![cIndex]!.includes(num) ? num : '' }}
                </div>
              </div>
            </div>
          </div>
        </div>
        </div>
        <div class="paused-overlay" v-if="isPaused">
          <h2>PAUSED</h2>
          <button class="game-btn primary-action" @click="resumeTimer">Resume</button>
        </div>
      </div>

      <div class="controls-container">
        <!-- Mobile Number Pad -->
        <div class="number-pad" v-if="isMobile">
          <button
            v-for="n in 9"
            :key="n"
            class="pad-btn"
            @mousedown.prevent
            @click="onPadInput(n)"
          >
            {{ n }}
          </button>
          <button class="pad-btn clear-btn" @mousedown.prevent @click="onPadClear">X</button>
        </div>

        <!-- Custom Mode Setup Controls -->
        <div class="controls" v-if="isDefiningCustom">
          <button class="game-btn" @click="startNewGame">Cancel</button>
          <div v-if="gameType === 'killer'" class="killer-controls">
            <button class="game-btn" v-if="!isCageSelectionMode" @click="toggleCageSelectionMode">Add Cage</button>
            <button class="game-btn delete-btn" v-if="!isCageSelectionMode && selectedCell && cages.some(c => c.cells.some(cl => cl.row === selectedCell!.r && cl.col === selectedCell!.c))" @click="deleteCage">Delete Cage</button>

            <div v-if="isCageSelectionMode" class="cage-actions">
              <span class="instruction">Select cells...</span>
              <button class="game-btn primary-action" @click="saveCage" :disabled="currentCageSelection.size === 0">Save Cage</button>
              <button class="game-btn" @click="cancelCageSelection">Cancel Selection</button>
            </div>
          </div>
          <button class="game-btn primary-action" v-if="!isCageSelectionMode" @click="validateAndStartCustom">Start Solving</button>
        </div>

        <!-- Normal Game Controls -->
        <div class="controls" v-else>
          <button class="game-btn" @click="startNewGame" :title="'New Game'">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            <span class="btn-text">New</span>
          </button>
          <button class="game-btn" @click="isPaused ? resumeTimer() : pauseTimer()" :title="isPaused ? 'Resume' : 'Pause'">
            <svg v-if="isPaused" class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            <svg v-else class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" />
            </svg>
            <span class="btn-text">{{ isPaused ? 'Resume' : 'Pause' }}</span>
          </button>
          <button class="game-btn" @click="undo" :disabled="isPaused || isGameOver" title="Undo">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 10h10a5 5 0 0 1 5 5v2M3 10l5-5M3 10l5 5" />
            </svg>
            <span class="btn-text">Undo</span>
          </button>
          <button class="game-btn" @click="shareGame" title="Share">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            <span class="btn-text">Share</span>
          </button>
          <button class="game-btn secondary" @click="showSolution" title="Show Solution">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
            </svg>
            <span class="btn-text">Solve</span>
          </button>
        </div>

        <div class="controls secondary-controls" v-if="!isDefiningCustom">
          <button
            class="game-btn"
            @click="isNoteMode = !isNoteMode"
            :class="{ 'active': isNoteMode }"
            title="Toggle Candidate Mode (Pencil)"
          >
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
            </svg>
            <span class="btn-text">{{ isNoteMode ? 'Notes ON' : 'Notes' }}</span>
          </button>
          <button class="game-btn" @click="generateCandidates" :class="['fill-notes-btn', { 'hidden-btn': candidatesPopulated }]" title="Populate Candidates">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
            </svg>
            <span class="btn-text">Fill Notes</span>
          </button>
          <button class="game-btn" @click="getHint" :disabled="isPaused || isGameOver" :class="['hint-btn', { 'hidden-btn': !candidatesPopulated }]" title="Get Hint">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z" />
            </svg>
            <span class="btn-text">Hint</span>
          </button>
          <button class="game-btn" @click="generateCandidates" :class="['refresh-btn', { 'hidden-btn': !candidatesPopulated }]" title="Refresh Candidates">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 4v6h-6M1 20v-6h6" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            <span class="btn-text">Refresh</span>
          </button>
        </div>
      </div>

      <!-- Feedback area - stable container for messages and hints -->
      <div class="feedback-area">
        <div v-if="message" class="message">{{ message }}</div>

        <!-- Hint Panel -->
        <div v-if="hintResult" class="hint-panel">
          <div class="hint-header">
            <strong>{{ hintResult.strategyName }}</strong>
            <span class="hint-difficulty">(Difficulty: {{ hintResult.difficulty.toFixed(1) }})</span>
          </div>
          <p class="hint-description">{{ hintResult.description }}</p>
          <div class="hint-actions">
            <button class="game-btn apply-btn" v-if="hintResult.eliminations.length > 0" @click="applyHint">
              Apply ({{ hintResult.eliminations.length }} elimination{{ hintResult.eliminations.length > 1 ? 's' : '' }})
            </button>
            <button class="game-btn dismiss-btn" @click="dismissHint">Dismiss</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.board-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.loading {
  font-size: 1.5rem;
  font-weight: bold;
  color: #374151;
  padding: 2rem;
}

:where(.dark, .dark *) .loading {
  color: #e5e7eb;
}

.header-status {
    text-align: center;
    margin-bottom: 1rem;
    color: #374151;
}

:where(.dark, .dark *) .header-status {
    color: #dad4f6;
}

.timer-display {
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    color: #374151;
    margin-bottom: 1rem;
    font-variant-numeric: tabular-nums;
}

:where(.dark, .dark *) .timer-display {
    color: #dad4f6;
}

.grid-outer {
  position: relative;
  width: 100%;
  max-width: 450px;
  margin-bottom: 20px;
}

.grid {
  display: table;
  table-layout: fixed;
  border-collapse: collapse;
  background-color: #000;
  border: 3px solid #000;
  width: 100%;
}

:where(.dark, .dark *) .grid {
  background-color: #1f2937;
  border-color: #374151;
}

.paused-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.85);
    z-index: 100;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
}

.row {
  display: table-row;
}

.cell {
  display: table-cell;
  background-color: white;
  border: 1px solid #ccc;
  box-sizing: border-box;
  vertical-align: top;
  padding: 0;
}

.cell-inner {
  position: relative;
  width: calc(450px / 9);
  height: calc(450px / 9);
}

:where(.dark, .dark *) .cell {
  background-color: #f3f4f6;
  border-color: #9ca3af;
}

/* Borders for 9x9 */
.grid.size-9 .cell:nth-child(3), 
.grid.size-9 .cell:nth-child(6) {
  border-right: 3px solid #000;
}
.grid.size-9 .row:nth-child(3) .cell, 
.grid.size-9 .row:nth-child(6) .cell {
  border-bottom: 3px solid #000;
}

/* Borders for 6x6 (2x3 blocks: 2 rows, 3 cols) */
/* Thick Right border after 3rd column */
.grid.size-6 .cell:nth-child(3) {
  border-right: 3px solid #000;
}
/* Thick Bottom border after 2nd and 4th row */
.grid.size-6 .row:nth-child(2) .cell,
.grid.size-6 .row:nth-child(4) .cell {
  border-bottom: 3px solid #000;
}

/* Killer Sudoku Cage Styling */
/* Use pseudo-element for cage borders to avoid conflict with grid lines */
.cell::after {
    content: '';
    position: absolute;
    top: 1px; left: 1px; right: 1px; bottom: 1px;
    pointer-events: none;
    z-index: 10;
    border: 0px dotted #555;
}

.cage-border-top::after { border-top-width: 2px; }
.cage-border-bottom::after { border-bottom-width: 2px; }
.cage-border-left::after { border-left-width: 2px; }
.cage-border-right::after { border-right-width: 2px; }

.cage-sum {
    position: absolute;
    top: 2px;
    left: 2px;
    font-size: 0.7rem;
    font-weight: bold;
    color: #333;
    z-index: 15;
    pointer-events: none;
    line-height: 1;
    background-color: white;
    padding: 0 2px;
    border-radius: 2px;
}

/* Input Styling */
.value-input {
  width: 100%;
  height: 100%;
  text-align: center;
  font-size: 1.5rem;
  border: none;
  outline: none;
  background: transparent;
  color: #007bff;
  font-weight: normal;
  position: absolute;
  top: 0;
  left: 0;
  z-index: 10;
  cursor: default;
}

.value-input.hidden {
    color: transparent;
    cursor: text;
}

.value-input.fixed {
    color: #000;
    font-weight: bold;
}

/* Highlight the active cell */
.cell:focus-within {
    background-color: rgba(187, 222, 251, 0.95) !important;
    z-index: 25;
    outline: 2px solid #007bff;
}

/* Incorrect Mark Styling */
.incorrect-mark {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: red;
    font-weight: bold;
    font-size: 1.5rem;
    pointer-events: none;
    z-index: 20;
    opacity: 0.8;
}

/* Candidates Grid */
.candidates-grid {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: grid;
    pointer-events: none;
    z-index: 5;
    box-sizing: border-box;
}

.candidates-grid.size-9 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
}

.candidates-grid.size-6 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
}

.candidates-grid.killer-mode {
    padding-top: 15px;
    padding-left: 5px;
    padding-right: 5px;
    padding-bottom: 5px;
}

.candidate-cell {
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 8px;
    color: #555;
    line-height: 1;
}

.controls-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 10px;
    width: 100%;
    max-width: 450px;
}

.controls {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap; /* Handle smaller screens */
}

.secondary-controls {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.secondary-controls .fill-notes-btn {
    grid-column: span 3;
}

.secondary-controls .hint-btn {
    grid-column: span 2;
}

.secondary-controls .refresh-btn {
    grid-column: span 1;
}

/* Game buttons - explicit class for Tailwind v4 specificity */
button.game-btn {
    padding: 8px 12px;
    font-size: 0.85rem;
    cursor: pointer;
    background-color: #42b983 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    min-width: auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

button.game-btn:hover {
    background-color: #3aa876 !important;
}

button.game-btn:disabled {
    background-color: #9ca3af !important;
    cursor: not-allowed;
    opacity: 0.6;
}

/* Number pad buttons */
button.pad-btn {
    padding: 12px;
    font-size: 1.2rem;
    font-weight: bold;
    cursor: pointer;
    background-color: #42b983 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    min-width: 44px;
    min-height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

button.pad-btn:hover {
    background-color: #3aa876 !important;
}

button.pad-btn.clear-btn {
    background-color: #e74c3c !important;
}

button.pad-btn.clear-btn:hover {
    background-color: #c0392b !important;
}

.btn-icon {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
}

.btn-text {
    white-space: nowrap;
}

button.game-btn.primary-action {
    background-color: #34495e !important;
    font-weight: bold;
}
button.game-btn.primary-action:hover {
    background-color: #2c3e50 !important;
}

select.difficulty-select {
    padding: 10px 15px;
    font-size: 0.9rem;
    cursor: pointer;
    background-color: #34495e;
    color: white;
    border: none;
    border-radius: 4px;
    outline: none;
}
select.difficulty-select:hover {
    background-color: #2c3e50;
}

button.game-btn.secondary {
    background-color: #e67e22 !important;
}
button.game-btn.secondary:hover {
    background-color: #d35400 !important;
}

button.game-btn.active {
    background-color: #34495e !important;
    border: 2px solid #dad4f6 !important;
}

.feedback-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 450px;
    margin-top: 0.5rem;
}

.message {
    font-weight: bold;
    font-size: 1.1rem;
    color: #374151;
    text-align: center;
}

:where(.dark, .dark *) .message {
    color: #dad4f6;
}

/* Hidden button class - removes from layout when not needed */
.hidden-btn {
    display: none;
}

.game-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}

.number-pad {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 10px;
    width: 100%;
    max-width: 450px;
}

.pad-btn {
    padding: 15px 0;
    font-size: 1.2rem;
    font-weight: bold;
    background-color: #f0f0f0;
    color: #333;
    border: 1px solid #ccc;
    border-radius: 8px;
    min-width: 0; /* Override default min-width */
}

.pad-btn:active {
    background-color: #e0e0e0;
}

.clear-btn {
    background-color: #ffcccc;
    color: #cc0000;
    grid-column: span 1;
}

.cage-selected {
    background-color: rgba(255, 235, 59, 0.5) !important;
}

.pointer-events-none {
    pointer-events: none;
}

.killer-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
}

.cage-actions {
    display: flex;
    gap: 5px;
    align-items: center;
}

.instruction {
    font-size: 0.9rem;
    color: #374151;
}

:where(.dark, .dark *) .instruction {
    color: #dad4f6;
}

.delete-btn {
    background-color: #e74c3c;
}
.delete-btn:hover {
    background-color: #c0392b;
}

/* Hint System Styles */
.hint-btn {
    background-color: #9b59b6;
}
.hint-btn:hover {
    background-color: #8e44ad;
}
.hint-btn:disabled {
    background-color: #bdc3c7;
    cursor: not-allowed;
}

.hint-panel {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    border: 2px solid #9b59b6;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
    width: 100%;
    box-sizing: border-box;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.hint-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    color: #fff;
}

.hint-difficulty {
    font-size: 0.85rem;
    color: #bdc3c7;
}

.hint-description {
    margin: 0.5rem 0;
    color: #ecf0f1;
    font-size: 0.95rem;
    line-height: 1.4;
}

.hint-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.apply-btn {
    background-color: #27ae60;
}
.apply-btn:hover {
    background-color: #219a52;
}

.dismiss-btn {
    background-color: #7f8c8d;
}
.dismiss-btn:hover {
    background-color: #6c7a7b;
}

/* Cell highlight classes for hints */
.hint-primary {
    background-color: rgba(39, 174, 96, 0.4) !important;
}

.hint-secondary {
    background-color: rgba(241, 196, 15, 0.3) !important;
}

.hint-elimination {
    background-color: rgba(231, 76, 60, 0.3) !important;
}

/* Candidate highlight classes */
.candidate-highlight-primary {
    color: #27ae60 !important;
    font-weight: bold;
}

.candidate-highlight-elimination {
    color: #e74c3c !important;
    text-decoration: line-through;
    font-weight: bold;
}

/* ========== Mobile Responsive Styles ========== */
@media (max-width: 480px) {
    .board-container {
        padding: 0;
    }

    .timer-display {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .grid {
        max-width: calc(100vw - 16px);
        margin-bottom: 10px;
    }

    .value-input {
        font-size: 1.2rem;
    }

    .cage-sum {
        font-size: 0.55rem;
    }

    .candidate-cell {
        font-size: 6px;
    }

    .candidates-grid.killer-mode {
        padding-top: 12px;
        padding-left: 3px;
        padding-right: 3px;
        padding-bottom: 3px;
    }

    .controls-container {
        gap: 6px;
        width: 100%;
        max-width: calc(100vw - 16px);
    }

    .controls {
        gap: 4px;
    }

    /* Hide button text on mobile, show only icons */
    .btn-text {
        display: none;
    }

    .btn-icon {
        width: 20px;
        height: 20px;
    }

    button {
        padding: 10px;
        min-width: 44px;
        min-height: 44px;
    }

    button.secondary {
        background-color: #e67e22;
    }

    .secondary-controls {
        margin-top: 4px;
        grid-template-columns: repeat(4, 1fr);
        gap: 4px;
    }

    .secondary-controls .fill-notes-btn {
        grid-column: span 3;
    }

    .secondary-controls .hint-btn {
        grid-column: span 2;
    }

    .secondary-controls .refresh-btn {
        grid-column: span 1;
    }

    .number-pad {
        max-width: calc(100vw - 16px);
        gap: 4px;
    }

    .pad-btn {
        padding: 12px 0;
        font-size: 1.1rem;
        min-height: 44px;
    }

    .feedback-area {
        max-width: calc(100vw - 16px);
    }

    .message {
        font-size: 0.9rem;
        padding: 0 8px;
    }

    .hint-panel {
        padding: 0.75rem;
    }

    .hint-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
    }

    .hint-description {
        font-size: 0.85rem;
    }

    .hint-actions {
        flex-wrap: wrap;
    }

    .hint-actions button {
        flex: 1;
        min-width: 80px;
    }

    .header-status h3 {
        font-size: 1.1rem;
        margin: 0;
    }

    .header-status p {
        font-size: 0.85rem;
        margin: 0.25rem 0;
    }

    .incorrect-mark {
        font-size: 1.2rem;
    }

    .paused-overlay h2 {
        font-size: 1.5rem;
    }

    .paused-overlay button {
        padding: 12px 24px;
    }

    .killer-controls {
        width: 100%;
        justify-content: center;
    }

    .cage-actions {
        flex-wrap: wrap;
        width: 100%;
        justify-content: center;
    }

    .instruction {
        font-size: 0.8rem;
        width: 100%;
        text-align: center;
    }
}

/* Small mobile screens (very small phones) */
@media (max-width: 360px) {
    .timer-display {
        font-size: 1.3rem;
    }

    .value-input {
        font-size: 1rem;
    }

    .candidate-cell {
        font-size: 5px;
    }

    .btn-icon {
        width: 18px;
        height: 18px;
    }

    button {
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
    }

    .pad-btn {
        padding: 10px 0;
        font-size: 1rem;
    }
}

/* Tablet and larger phone landscape */
@media (min-width: 481px) and (max-width: 768px) {
    .grid {
        max-width: 400px;
    }

    .controls {
        gap: 8px;
    }

    button {
        padding: 8px 10px;
    }

    .btn-text {
        font-size: 0.8rem;
    }
}

/* Dark mode overrides */
:where(.dark, .dark *) .pad-btn {
    background-color: #374151;
    color: #f3f4f6;
    border-color: #4b5563;
}

:where(.dark, .dark *) .pad-btn:active {
    background-color: #4b5563;
}

:where(.dark, .dark *) .clear-btn {
    background-color: #7f1d1d;
    color: #fecaca;
}

:where(.dark, .dark *) .cage-sum {
    background-color: #e5e7eb;
}

:where(.dark, .dark *) .value-input {
    color: #2563eb;
}

:where(.dark, .dark *) .value-input.fixed {
    color: #111827;
}

:where(.dark, .dark *) .candidate-cell {
    color: #6b7280;
}
</style>
