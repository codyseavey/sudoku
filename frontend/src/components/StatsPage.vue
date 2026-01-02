<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits(['back-to-menu'])

const stats = ref({ totalSolved: 0 })
const loading = ref(true)

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
            <div class="stat-item">
                <span class="label">Total Puzzles Solved:</span>
                <span class="value">{{ stats.totalSolved }}</span>
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
    text-align: center;
}

h2 {
    margin: 0;
    color: #2c3e50;
}

.stat-item {
    font-size: 1.2rem;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}

.label {
    font-weight: bold;
}

.back-btn {
    padding: 1rem;
    background: #34495e;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
}

.back-btn:hover {
    background: #2c3e50;
}
</style>
