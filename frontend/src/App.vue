<script setup lang="ts">
import { ref, onMounted } from 'vue'
import SudokuBoard from './components/SudokuBoard.vue'
import LandingPage from './components/LandingPage.vue'
import StatsPage from './components/StatsPage.vue'
import PuzzleSolved from './components/PuzzleSolved.vue'

type ViewState = 'landing' | 'game' | 'stats' | 'solved'
const currentView = ref<ViewState>('landing')
const solvedBoard = ref<(number | null)[][]>([])

const gameSettings = ref({ difficulty: 'medium', size: 9, isCustomMode: false, gameType: 'standard' })
const importedPuzzle = ref<Record<string, any> | undefined>(undefined)

const onStartGame = (settings: { difficulty: string, size: number, gameType?: string }) => {
    gameSettings.value = { ...settings, isCustomMode: false, gameType: settings.gameType || 'standard' }
    importedPuzzle.value = undefined
    currentView.value = 'game'
}

const onCreateCustom = (settings: { size: number, gameType?: string }) => {
    gameSettings.value = { difficulty: 'custom', size: settings.size, isCustomMode: true, gameType: settings.gameType || 'standard' }
    importedPuzzle.value = undefined
    currentView.value = 'game'
}

const onBackToMenu = () => {
    // Clear any saved game state
    localStorage.removeItem(`sudoku_game_state_${gameSettings.value.size}`)
    // Also try clearing the other size just in case
    localStorage.removeItem(`sudoku_game_state_6`)
    localStorage.removeItem(`sudoku_game_state_9`)
    
    // Clear URL params
    window.history.replaceState({}, document.title, "/")
    
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

onMounted(() => {
    // Check for import param
    const params = new URLSearchParams(window.location.search)
    const importData = params.get('import')
    
    if (importData) {
        try {
            const decoded = JSON.parse(atob(importData))
            if (decoded.board && decoded.solution) {
                gameSettings.value = { 
                    difficulty: decoded.difficulty || 'imported', 
                    size: decoded.size || 9,
                    isCustomMode: false,
                    gameType: 'standard'
                }
                importedPuzzle.value = decoded
                currentView.value = 'game'
                return
            }
        } catch (e) {
            console.error("Failed to parse imported puzzle", e)
        }
    }

    // Check if there is an active game (only if no import)
    // We check both potential keys
    const saved9 = localStorage.getItem('sudoku_game_state_9')
    const saved6 = localStorage.getItem('sudoku_game_state_6')
    
    if (saved9) {
        try {
            const state = JSON.parse(saved9)
            if (state && state.board) {
                gameSettings.value = { 
                    difficulty: state.difficulty || 'medium', 
                    size: 9,
                    isCustomMode: state.isCustomMode || false,
                    gameType: state.gameType || 'standard'
                }
                currentView.value = 'game'
                return
            }
        } catch (e) {}
    }
    
    if (saved6) {
        try {
            const state = JSON.parse(saved6)
            if (state && state.board) {
                gameSettings.value = { 
                    difficulty: state.difficulty || 'medium', 
                    size: 6,
                    isCustomMode: state.isCustomMode || false,
                    gameType: state.gameType || 'standard'
                }
                currentView.value = 'game'
                return
            }
        } catch (e) {}
    }
})
</script>

<template>
  <div class="container">
    <h1 v-if="currentView === 'landing'">Sudoku</h1>

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
        :gameType="gameSettings.gameType"
        @back-to-menu="onBackToMenu"
        @new-game="onStartGame(gameSettings)"
    />

    <SudokuBoard 
        v-else-if="currentView === 'game'"
        :initialDifficulty="gameSettings.difficulty" 
        :size="gameSettings.size"
        :isCustomMode="gameSettings.isCustomMode"
        :gameType="gameSettings.gameType"
        :initialPuzzle="importedPuzzle"
        @back-to-menu="onBackToMenu"
        @puzzle-completed="onPuzzleCompleted"
    />
  </div>
</template>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: Arial, sans-serif;
  margin-top: 2rem;
}

h1 {
  color: #dad4f6;
}
</style>
