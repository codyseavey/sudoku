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
    gap: 1rem;
    margin-top: 1rem;
    color: #dad4f6;
    width: 100%;
    padding: 0 0.5rem;
    box-sizing: border-box;
}

.header {
    text-align: center;
}

.header h1 {
    font-size: 1.8rem;
    margin: 0 0 0.5rem 0;
}

.header p {
    font-size: 1rem;
    margin: 0;
}

.grid {
  display: grid;
  grid-template-columns: repeat(v-bind(size), 1fr);
  background-color: #000;
  border: 3px solid #000;
  gap: 0;
  width: 100%;
  max-width: 280px;
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
  font-size: 1.1rem;
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
    gap: 0.75rem;
    width: 100%;
    max-width: 280px;
}

button {
    padding: 0.85rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    border: none;
    border-radius: 4px;
    color: white;
    flex: 1;
    min-height: 48px;
    touch-action: manipulation;
}

.menu-btn {
    background-color: #34495e;
}

.menu-btn:hover {
    background-color: #2c3e50;
}

.new-btn {
    background-color: #42b983;
}

.new-btn:hover {
    background-color: #3aa876;
}

/* Mobile styles */
@media (max-width: 480px) {
    .solved-container {
        gap: 0.75rem;
        margin-top: 0.5rem;
        padding: 0 0.25rem;
    }

    .header h1 {
        font-size: 1.5rem;
    }

    .header p {
        font-size: 0.9rem;
    }

    .grid {
        max-width: min(280px, calc(100vw - 2rem));
    }

    .cell {
        font-size: 0.9rem;
    }

    .buttons {
        max-width: min(280px, calc(100vw - 2rem));
        gap: 0.5rem;
    }

    button {
        padding: 0.7rem 0.5rem;
        font-size: 0.9rem;
        min-height: 44px;
    }
}

/* Very small screens */
@media (max-width: 360px) {
    .header h1 {
        font-size: 1.3rem;
    }

    .cell {
        font-size: 0.8rem;
    }

    button {
        font-size: 0.85rem;
    }
}
</style>
