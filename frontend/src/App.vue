<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useThemeStore } from './stores/theme'
import SudokuBoard from './components/SudokuBoard.vue'
import LandingPage from './components/LandingPage.vue'
import StatsPage from './components/StatsPage.vue'
import PuzzleSolved from './components/PuzzleSolved.vue'

const themeStore = useThemeStore()

type ViewState = 'landing' | 'game' | 'stats' | 'solved'
const currentView = ref<ViewState>('landing')
const solvedBoard = ref<(number | null)[][]>([])

const gameSettings = ref({ difficulty: 'medium', size: 9, isCustomMode: false, gameType: 'standard' })
const importedPuzzle = ref<Record<string, any> | undefined>(undefined)
// For raw board data from upload (array of arrays)
const initialBoardState = ref<number[][] | undefined>(undefined)
const initialCagesState = ref<{ sum: number, cells: { row: number, col: number }[] }[] | undefined>(undefined)

const onStartGame = (settings: { difficulty: string, size: number, gameType?: string }) => {
    gameSettings.value = { ...settings, isCustomMode: false, gameType: settings.gameType || 'standard' }
    importedPuzzle.value = undefined
    initialBoardState.value = undefined
    initialCagesState.value = undefined
    currentView.value = 'game'
}

const onCreateCustom = (settings: { size: number, gameType?: string, initialBoard?: number[][], initialCages?: { sum: number, cells: { row: number, col: number }[] }[] }) => {
    gameSettings.value = { difficulty: 'custom', size: settings.size, isCustomMode: true, gameType: settings.gameType || 'standard' }
    importedPuzzle.value = undefined
    initialBoardState.value = settings.initialBoard
    initialCagesState.value = settings.initialCages
    currentView.value = 'game'
}

const onBackToMenu = () => {
    // Clear any saved game state
    localStorage.removeItem(`sudoku_game_state_${gameSettings.value.size}`)
    // Also try clearing the other size just in case
    localStorage.removeItem('sudoku_game_state_6')
    localStorage.removeItem('sudoku_game_state_9')
    
    // Clear URL params
    window.history.replaceState({}, document.title, '/')
    
    currentView.value = 'landing'
    importedPuzzle.value = undefined
}

const onViewStats = () => {
    currentView.value = 'stats'
}

const onPuzzleCompleted = (data: { board: (number | null)[][] }) => {
    solvedBoard.value = data.board
    currentView.value = 'solved'

    // Clear saved game since it is solved
    localStorage.removeItem(`sudoku_game_state_${gameSettings.value.size}`)
}

// Storage key constant
const STORAGE_KEY_PREFIX = 'sudoku_game_state_'

// Helper to load saved game state for a given size
interface SavedGameState {
    board: (number | null)[][]
    difficulty?: string
    isCustomMode?: boolean
    gameType?: string
}

const loadSavedGameState = (size: number): SavedGameState | null => {
    const saved = localStorage.getItem(`${STORAGE_KEY_PREFIX}${size}`)
    if (!saved) return null

    try {
        const state = JSON.parse(saved)
        if (state?.board) {
            return state as SavedGameState
        }
    } catch (e) {
        console.error(`Failed to parse saved game state for size ${size}`, e)
    }
    return null
}

// Helper to parse imported puzzle from URL
const parseImportedPuzzle = (importData: string) => {
    try {
        const decoded = JSON.parse(atob(importData))
        if (decoded.board && decoded.solution) {
            return decoded
        }
    } catch (e) {
        console.error('Failed to parse imported puzzle', e)
    }
    return null
}

// Theme icon paths
const themeIcon = () => {
    if (themeStore.currentTheme === 'dark') return 'M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z'
    if (themeStore.currentTheme === 'light') return 'M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z'
    return 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z'
}

onMounted(() => {
    // Initialize theme
    themeStore.initTheme()
    
    // Check for import param first
    const params = new URLSearchParams(window.location.search)
    const importData = params.get('import')

    if (importData) {
        const decoded = parseImportedPuzzle(importData)
        if (decoded) {
            gameSettings.value = {
                difficulty: decoded.difficulty || 'imported',
                size: decoded.size || 9,
                isCustomMode: false,
                gameType: decoded.gameType || 'standard'
            }
            importedPuzzle.value = decoded
            // For Killer Sudoku imports, also set the cages
            if (decoded.cages && decoded.cages.length > 0) {
                initialCagesState.value = decoded.cages
            }
            currentView.value = 'game'
            return
        }
    }

    // Check for saved games (9x9 first, then 6x6)
    for (const size of [9, 6]) {
        const state = loadSavedGameState(size)
        if (state) {
            gameSettings.value = {
                difficulty: state.difficulty || 'medium',
                size,
                isCustomMode: state.isCustomMode || false,
                gameType: state.gameType || 'standard'
            }
            currentView.value = 'game'
            return
        }
    }
})
</script>

<template>
  <div class="min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors">
    <!-- Theme toggle button (fixed position) -->
    <button
      @click="themeStore.cycleTheme()"
      class="fixed top-4 right-4 p-2 rounded-lg bg-white dark:bg-gray-800 shadow-md hover:shadow-lg transition-shadow z-50"
      :title="`Theme: ${themeStore.currentTheme}`"
    >
      <svg class="h-5 w-5 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="themeIcon()" />
      </svg>
    </button>

    <div class="flex flex-col items-center w-full max-w-full box-border pt-4 px-2 sm:px-4">
      <h1 v-if="currentView === 'landing'" class="text-gray-700 dark:text-gray-100 text-4xl sm:text-5xl font-bold my-2">
        Sudoku
      </h1>

      <LandingPage
        v-if="currentView === 'landing'"
        @start-game="onStartGame"
        @create-custom="onCreateCustom"
        @view-stats="onViewStats"
      />

      <StatsPage
        v-else-if="currentView === 'stats'"
        @back-to-menu="onBackToMenu"
      />

      <PuzzleSolved
        v-else-if="currentView === 'solved'"
        :board="solvedBoard"
        :difficulty="gameSettings.difficulty"
        :size="gameSettings.size"
        :game-type="gameSettings.gameType"
        @back-to-menu="onBackToMenu"
        @new-game="onStartGame(gameSettings)"
      />

      <SudokuBoard 
        v-else-if="currentView === 'game'"
        :initial-difficulty="gameSettings.difficulty" 
        :size="gameSettings.size"
        :is-custom-mode="gameSettings.isCustomMode"
        :game-type="gameSettings.gameType"
        :initial-puzzle="importedPuzzle"
        :initial-board-upload="initialBoardState"
        :initial-cages="initialCagesState"
        @back-to-menu="onBackToMenu"
        @puzzle-completed="onPuzzleCompleted"
      />
    </div>
  </div>
</template>
