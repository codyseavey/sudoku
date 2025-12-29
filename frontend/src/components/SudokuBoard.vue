<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

// Use null for empty cells instead of 0
const board = ref<(number | null)[][]>(Array.from({ length: 9 }, () => Array(9).fill(null)))
const isFixed = ref<boolean[][]>(Array.from({ length: 9 }, () => Array(9).fill(false)))
const solution = ref<number[][]>([])
const candidates = ref<number[][][]>(Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => [])))
// Track candidates that have been explicitly removed (manually or via auto-eliminate)
// We need to serialize Sets to Arrays for storage
const eliminatedCandidates = ref<Set<number>[][]>(Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => new Set())))

// History Stack for Undo
interface GameState {
    board: (number | null)[][];
    candidates: number[][][];
    eliminatedCandidates: Set<number>[][];
}
const history = ref<GameState[]>([])

const loading = ref(false)
const message = ref('')
const isNoteMode = ref(false)
const difficulty = ref('easy')

const STORAGE_KEY = 'sudoku_game_state'

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
    if (history.value.length === 0) return
    const prevState = history.value.pop()
    if (prevState) {
        board.value = prevState.board
        candidates.value = prevState.candidates
        eliminatedCandidates.value = prevState.eliminatedCandidates
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
        candidates: candidates.value,
        // Convert Sets to Arrays for JSON serialization
        eliminatedCandidates: eliminatedCandidates.value.map(row => 
            row.map(set => Array.from(set))
        ),
        difficulty: difficulty.value,
        history: serializeHistory(history.value)
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(gameState))
}

const loadGame = (): boolean => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
        try {
            const gameState = JSON.parse(saved)
            board.value = gameState.board
            isFixed.value = gameState.isFixed
            solution.value = gameState.solution
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
            return true
        } catch (e) {
            console.error("Failed to load game state", e)
            return false
        }
    }
    return false
}

const startNewGame = () => {
    localStorage.removeItem(STORAGE_KEY)
    history.value = [] // Clear history on new game
    fetchPuzzle()
}

const fetchPuzzle = async () => {
  loading.value = true
  message.value = ''
  try {
    const response = await fetch(`/api/puzzle?difficulty=${difficulty.value}`)
    const data = await response.json()
    // Convert 0s to nulls for the UI
    board.value = data.board.map((row: number[]) => row.map(val => val === 0 ? null : val))
    // Mark non-empty cells as fixed
    isFixed.value = data.board.map((row: number[]) => row.map(val => val !== 0))
    
    solution.value = data.solution
    // Reset candidates and elimination history
    resetCandidates()
    history.value = [] // Reset history
    saveGame() // Save initial state
  } catch (error) {
    console.error('Error fetching puzzle:', error)
    message.value = 'Failed to load puzzle.'
  } finally {
    loading.value = false
  }
}

const resetCandidates = () => {
    candidates.value = Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => []))
    eliminatedCandidates.value = Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => new Set()))
}

// Watch for changes to save state
watch([board, candidates, eliminatedCandidates, difficulty], () => {
    saveGame()
}, { deep: true })

const checkSolution = () => {
    if (solution.value.length === 0) return

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const val = board.value[i]![j]
            // Treat null as incorrect/incomplete
            if (val === null || val !== solution.value[i]![j]) {
                message.value = 'Incorrect or incomplete! Keep trying.'
                return
            }
        }
    }
    message.value = 'Correct! Well done.'
}

const isBoardFull = () => {
    return board.value.every(row => row.every(cell => cell !== null))
}

const showSolution = () => {
    if (solution.value.length === 0) return
    saveState() // Save before revealing
    board.value = solution.value.map(row => [...row])
    resetCandidates()
    message.value = 'Solution revealed.'
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

    // Row
    for (let j = 0; j < 9; j++) removeFromCell(r, j, val)
    // Col
    for (let i = 0; i < 9; i++) removeFromCell(i, c, val)
    // Box
    const startRow = Math.floor(r / 3) * 3
    const startCol = Math.floor(c / 3) * 3
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            removeFromCell(startRow + i, startCol + j, val)
        }
    }
}

// Interaction Handler
const handleInputKey = (e: KeyboardEvent, r: number, c: number) => {
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
            if (!isFixed.value[r]![c] && board.value[r]![c] !== null) {
                 saveState() // Save before clearing
                 board.value[r]![c] = null
            }
        }
        return
    }

    // Handle numbers 1-9
    if (/^[1-9]$/.test(key)) {
        e.preventDefault() // We handle the value manually
        
        // Prevent editing if fixed
        if (isFixed.value[r]![c]) return

        const num = parseInt(key)
        
        // Save state before modification
        saveState()

        if (isNoteMode.value) {
            // Toggle candidate
            const currentCandidates = candidates.value[r]![c]!
            const idx = currentCandidates.indexOf(num)
            if (idx > -1) {
                // Removing a candidate manually
                currentCandidates.splice(idx, 1)
                eliminatedCandidates.value[r]![c]!.add(num)
            } else {
                // Adding a candidate manually (override elimination)
                currentCandidates.push(num)
                currentCandidates.sort()
                eliminatedCandidates.value[r]![c]!.delete(num)
            }
        } else {
            // Set value
            board.value[r]![c] = num
            // Clear candidates for this cell when a value is set
            candidates.value[r]![c] = []
            
            // Auto Eliminate Peer Candidates (Always ON)
            // Only eliminate if the number is correct (matches solution)
            if (num === solution.value[r]![c]) {
                removeCandidates(r, c, num)
            }

            // Auto Check if full
            if (isBoardFull()) {
                checkSolution()
            }
        }
    } else {
        // Block other keys
        e.preventDefault()
    }
}

// Auto Candidate Logic
const generateCandidates = () => {
    saveState() // Save before generating
    for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
            if (board.value[r]![c] === null) {
                const valid = getValidNumbers(r, c)
                // Filter out candidates that have been explicitly eliminated
                candidates.value[r]![c] = valid.filter(n => !eliminatedCandidates.value[r]![c]!.has(n))
            } else {
                candidates.value[r]![c] = []
            }
        }
    }
    message.value = 'Candidates generated (respecting eliminations).'
}

const getValidNumbers = (row: number, col: number): number[] => {
    const used = new Set<number>()
    
    // Check Row
    for (let c = 0; c < 9; c++) {
        const val = board.value[row]![c]
        if (val !== null && val !== undefined) used.add(val)
    }

    // Check Col
    for (let r = 0; r < 9; r++) {
        const val = board.value[r]![col]
        if (val !== null && val !== undefined) used.add(val)
    }

    // Check Box
    const startRow = Math.floor(row / 3) * 3
    const startCol = Math.floor(col / 3) * 3
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const val = board.value[startRow + r]![startCol + c]
            if (val !== null && val !== undefined) used.add(val)
        }
    }

    const valid: number[] = []
    for (let n = 1; n <= 9; n++) {
        if (!used.has(n)) valid.push(n)
    }
    return valid
}

onMounted(() => {
    if (!loadGame()) {
        fetchPuzzle()
    }
})
</script>

<template>
  <div class="board-container">
    <div v-if="loading" class="loading">Generating Puzzle...</div>
    
    <div v-else class="game-area">
        <div class="grid">
        <div v-for="(row, rIndex) in board" :key="rIndex" class="row">
            <div v-for="(cell, cIndex) in row" :key="cIndex" class="cell">
                <!-- Main Value Input -->
                <input 
                    type="text"
                    inputmode="numeric"
                    :value="cell"
                    @keydown="handleInputKey($event, rIndex, cIndex)"
                    readonly
                    class="value-input"
                    :class="{ 
                        'hidden': cell === null,
                        'fixed': isFixed[rIndex]![cIndex] 
                    }"
                />
                
                <!-- Incorrect Mark (Red X) -->
                <div v-if="!isFixed[rIndex]![cIndex] && cell !== null && cell !== solution[rIndex]![cIndex]" class="incorrect-mark">
                    X
                </div>

                <!-- Candidates Overlay -->
                <div v-if="cell === null && candidates[rIndex]![cIndex]!.length > 0" class="candidates-grid">
                    <div 
                        v-for="num in 9" 
                        :key="num" 
                        class="candidate-cell"
                    >
                        {{ candidates[rIndex]![cIndex]!.includes(num) ? num : '' }}
                    </div>
                </div>
            </div>
        </div>
        </div>

        <div class="controls-container">
            <div class="controls">
                <select v-model="difficulty" class="difficulty-select">
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                </select>
                <button @click="startNewGame">New Game</button>
                <button @click="undo">Undo</button>
                <button @click="showSolution" class="secondary">Show Solution</button>
            </div>
            
            <div class="controls secondary-controls">
                <button 
                    @click="isNoteMode = !isNoteMode" 
                    :class="{ 'active': isNoteMode }"
                    title="Toggle Note Mode"
                >
                    Note Mode: {{ isNoteMode ? 'ON' : 'OFF' }}
                </button>
                <button @click="generateCandidates">Auto Candidates</button>
            </div>
        </div>

        <div v-if="message" class="message">{{ message }}</div>
    </div>
  </div>
</template>

<style scoped>
.board-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.grid {
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  background-color: #000;
  border: 3px solid #000;
  margin-bottom: 20px;
  gap: 0;
  width: 100%;
  max-width: 450px; /* Adjust based on desired size */
}

.row {
  display: contents;
}

.cell {
  background-color: white;
  aspect-ratio: 1;
  position: relative;
  border: 1px solid #ccc;
  box-sizing: border-box;
}

/* Thick Vertical Lines */
.cell:nth-child(3), .cell:nth-child(6) {
  border-right: 3px solid #000;
}

/* Thick Horizontal Lines */
.row:nth-child(3) .cell, .row:nth-child(6) .cell {
  border-bottom: 3px solid #000;
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

.value-input.fixed {
    color: #000;
    font-weight: bold;
}

.value-input.hidden {
    opacity: 0;
    cursor: text;
}

.value-input:focus {
    background-color: rgba(66, 185, 131, 0.1);
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
    pointer-events: none; /* Allows clicks to pass through to input */
    z-index: 20; /* Above input */
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
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
    pointer-events: none;
    z-index: 5;
}

.candidate-cell {
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 10px;
    color: #555;
    line-height: 1;
}

.controls-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 10px;
}

.controls {
    display: flex;
    gap: 10px;
    justify-content: center;
}

button {
    padding: 10px 15px;
    font-size: 0.9rem;
    cursor: pointer;
    background-color: #42b983;
    color: white;
    border: none;
    border-radius: 4px;
    min-width: 80px;
}

button:hover {
    background-color: #3aa876;
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

button.secondary {
    background-color: #e67e22;
}
button.secondary:hover {
    background-color: #d35400;
}

button.active {
    background-color: #34495e;
    border: 2px solid #dad4f6;
}

.message {
    font-weight: bold;
    font-size: 1.1rem;
    color: #dad4f6;
}
</style>
