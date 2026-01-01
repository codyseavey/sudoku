<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits(['start-game', 'create-custom'])

const difficulty = ref('medium')
const size = ref(9)
const gameType = ref('standard')

const startGame = () => {
  emit('start-game', { difficulty: difficulty.value, size: size.value, gameType: gameType.value })
}
</script>

<template>
  <div class="landing-container">
    <div class="card">
        <h2>New Game Settings</h2>
        
        <div class="setting-group">
            <label>Game Type:</label>
            <div class="options">
                <button
                    :class="{ active: gameType === 'standard' }"
                    @click="gameType = 'standard'"
                >
                    Standard
                </button>
                <button
                    :class="{ active: gameType === 'killer' }"
                    @click="gameType = 'killer'"
                >
                    Killer Sudoku
                </button>
            </div>
        </div>

        <div class="setting-group">
            <label>Grid Size:</label>
            <div class="options">
                <button 
                    :class="{ active: size === 9 }" 
                    @click="size = 9"
                >
                    Standard (9x9)
                </button>
                <button 
                    :class="{ active: size === 6 }" 
                    @click="size = 6"
                >
                    Mini (6x6)
                </button>
            </div>
        </div>

        <div class="setting-group">
            <label>Difficulty:</label>
            <div class="options">
                <button 
                    :class="{ active: difficulty === 'easy' }" 
                    @click="difficulty = 'easy'"
                >
                    Easy
                </button>
                <button 
                    :class="{ active: difficulty === 'medium' }" 
                    @click="difficulty = 'medium'"
                >
                    Medium
                </button>
                <button 
                    :class="{ active: difficulty === 'hard' }" 
                    @click="difficulty = 'hard'"
                >
                    Hard
                </button>
            </div>
        </div>

        <div class="buttons">
            <button class="start-btn" @click="startGame">Start Game</button>
            <button
                class="custom-btn"
                @click="emit('create-custom', { size, gameType })"
                :disabled="gameType === 'killer'"
                :title="gameType === 'killer' ? 'Custom game creation is not supported for Killer Sudoku' : ''"
            >
                Create Custom
            </button>
        </div>
    </div>
  </div>
</template>

<style scoped>
.landing-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
}

.card {
    background: #fff;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: #333;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    width: 100%;
    max-width: 400px;
}

h2 {
    margin: 0;
    text-align: center;
    color: #2c3e50;
}

.setting-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

label {
    font-weight: bold;
    font-size: 0.9rem;
}

.options {
    display: flex;
    gap: 0.5rem;
}

.options button {
    flex: 1;
    padding: 0.5rem;
    border: 1px solid #ccc;
    background: #f8f9fa;
    color: #333;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
}

.options button.active {
    background: #42b983;
    color: white;
    border-color: #42b983;
    font-weight: bold;
}

.buttons {
    display: flex;
    gap: 1rem;
}

.start-btn, .custom-btn {
    flex: 1;
    padding: 1rem;
    color: white;
    font-size: 1.1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 1rem;
}

.start-btn {
    background: #34495e;
}
.start-btn:hover {
    background: #2c3e50;
}

.custom-btn {
    background: #e67e22;
}
.custom-btn:hover {
    background: #d35400;
}
.custom-btn:disabled {
    background: #95a5a6;
    cursor: not-allowed;
}
.custom-btn:disabled:hover {
    background: #95a5a6;
}
</style>
