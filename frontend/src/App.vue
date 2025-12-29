<script setup lang="ts">
import SudokuBoard from './components/SudokuBoard.vue'
</script>

<template>
  <div class="container">
    <h1>Sudoku</h1>
    <SudokuBoard />
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
