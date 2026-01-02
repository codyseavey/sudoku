<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits(['back-to-menu'])

const stats = ref({ totalSolved: 0, details: {} as Record<string, Record<string, number>> })
const loading = ref(true)

const formatType = (type: string) => {
    if (type === 'standard') return 'Classic Sudoku'
    if (type === 'killer') return 'Killer Sudoku'
    return type.charAt(0).toUpperCase() + type.slice(1)
}

const formatDifficulty = (diff: string) => {
    return diff.charAt(0).toUpperCase() + diff.slice(1)
}

onMounted(async () => {
    try {
        const res = await fetch('/api/stats')
        stats.value = await res.json()
    } catch (e) {
        console.error("Failed to load stats", e)
    } finally {
        loading.value = false
    }
})
</script>

<template>
  <div class="stats-container">
    <div class="card">
        <h2>Global Stats</h2>
        <div v-if="loading">Loading...</div>
        <div v-else class="stats-content">
            <div class="stat-item main-stat">
                <span class="label">Total Puzzles Solved:</span>
                <span class="value">{{ stats.totalSolved }}</span>
            </div>

            <div v-for="(difficulties, type) in stats.details" :key="type" class="type-section">
                <h3>{{ formatType(type) }}</h3>
                <div v-for="(count, difficulty) in difficulties" :key="difficulty" class="stat-item sub-stat">
                    <span class="label">{{ formatDifficulty(difficulty) }}:</span>
                    <span class="value">{{ count }}</span>
                </div>
            </div>
        </div>
        <button class="back-btn" @click="emit('back-to-menu')">Back to Menu</button>
    </div>
  </div>
</template>

<style scoped>
.stats-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    margin-top: 2rem;
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

h3 {
    margin: 1rem 0 0.5rem 0;
    font-size: 1.1rem;
    color: #34495e;
    border-bottom: 2px solid #dad4f6;
    padding-bottom: 0.2rem;
}

.stat-item {
    font-size: 1rem;
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
}

.stat-item.main-stat {
    font-size: 1.2rem;
    font-weight: bold;
    border-bottom: 1px solid #eee;
    margin-bottom: 1rem;
}

.stat-item.sub-stat {
    padding-left: 1rem;
    border-bottom: 1px solid #f9f9f9;
}

.label {
    font-weight: 500;
}

.back-btn {
    padding: 1rem;
    background: #34495e;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    margin-top: 1rem;
}

.back-btn:hover {
    background: #2c3e50;
}
</style>
