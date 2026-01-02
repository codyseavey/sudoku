<script setup lang="ts">
import type { PropType } from 'vue'

defineProps({
    board: {
        type: Array as PropType<(number | null)[][]>,
        required: true
    },
    difficulty: {
        type: String,
        required: true
    },
    size: {
        type: Number,
        required: true
    },
    gameType: {
        type: String,
        default: 'standard'
    }
})

const emit = defineEmits(['new-game', 'back-to-menu'])

</script>

<template>
  <div class="solved-container">
    <div class="header">
        <h1>Puzzle Solved!</h1>
        <p>Great job completing this {{ difficulty }} puzzle.</p>
    </div>

    <!-- We reuse SudokuBoard but can pass a prop to make it read-only or just rely on it being solved -->
    <!-- Ideally, we just render the grid. Since SudokuBoard has logic we might not want,
         we can either create a simple grid renderer OR reuse SudokuBoard in a "review" mode.
         For simplicity and consistency, let's reuse SudokuBoard but maybe we can just pass the data as an "imported" puzzle
         and somehow disable interaction.
         However, SudokuBoard is complex. Let's just create a simple read-only view here to avoid side effects.
    -->
    <div class="grid" :class="`size-${size}`">
        <div v-for="(row, rIndex) in board" :key="rIndex" class="row">
            <div v-for="(cell, cIndex) in row" :key="cIndex" class="cell">
                <span class="value">{{ cell }}</span>
            </div>
        </div>
    </div>

    <div class="buttons">
        <button class="menu-btn" @click="emit('back-to-menu')">Back to Menu</button>
        <button class="new-btn" @click="emit('new-game')">New Game</button>
    </div>
  </div>
</template>

<style scoped>
.solved-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    margin-top: 2rem;
    color: #dad4f6;
}

.header {
    text-align: center;
}

.grid {
  display: grid;
  grid-template-columns: repeat(v-bind(size), 1fr);
  background-color: #000;
  border: 3px solid #000;
  gap: 0;
  width: 100%;
  max-width: 300px; /* Smaller preview */
}

.row {
  display: contents;
}

.cell {
  background-color: white;
  aspect-ratio: 1;
  border: 1px solid #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 1.2rem;
  color: #000;
  font-weight: bold;
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

/* Borders for 6x6 */
.grid.size-6 .cell:nth-child(3) {
  border-right: 3px solid #000;
}
.grid.size-6 .row:nth-child(2) .cell,
.grid.size-6 .row:nth-child(4) .cell {
  border-bottom: 3px solid #000;
}

.buttons {
    display: flex;
    gap: 1rem;
}

button {
    padding: 1rem 1.5rem;
    font-size: 1.1rem;
    cursor: pointer;
    border: none;
    border-radius: 4px;
    color: white;
}

.menu-btn {
    background-color: #34495e;
}

.new-btn {
    background-color: #42b983;
}
</style>
