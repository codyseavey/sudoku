<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits(['back-to-menu'])

const stats = ref({ totalSolved: 0, details: {} as Record<string, Record<string, Record<string, number>>> })
const loading = ref(true)

const formatType = (type: string) => {
    if (type === 'standard') return 'Classic Sudoku'
    if (type === 'killer') return 'Killer Sudoku'
    return type.charAt(0).toUpperCase() + type.slice(1)
}

const formatSize = (size: string) => {
    return `${size}x${size}`
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

            <div v-for="(sizes, type) in stats.details" :key="type" class="type-section">
                <h3>{{ formatType(type) }}</h3>
                <div v-for="(difficulties, size) in sizes" :key="size" class="size-section">
                    <!-- Guard against malformed/legacy data where size might be a difficulty key -->
                    <template v-if="typeof difficulties === 'object' && !isNaN(parseInt(size))">
                        <h4>{{ formatSize(size) }}</h4>
                        <div v-for="(count, difficulty) in difficulties" :key="difficulty" class="stat-item sub-stat">
                            <span class="label">{{ formatDifficulty(difficulty) }}:</span>
                            <span class="value">{{ count }}</span>
                        </div>
                    </template>
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
    align-items: flex-start;
    width: 100%;
    padding: 0 0.5rem;
    box-sizing: border-box;
    margin-top: 1rem;
}

.card {
    background: #fff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: #333;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    width: 100%;
    max-width: 400px;
    box-sizing: border-box;
}

h2 {
    margin: 0;
    text-align: center;
    color: #2c3e50;
    font-size: 1.3rem;
}

h3 {
    margin: 1rem 0 0.4rem 0;
    font-size: 1.1rem;
    color: #34495e;
    border-bottom: 2px solid #dad4f6;
    padding-bottom: 0.2rem;
    text-align: left;
}

h4 {
    margin: 0.4rem 0 0.2rem 0.75rem;
    font-size: 0.9rem;
    color: #7f8c8d;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-item {
    font-size: 0.95rem;
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
}

.stat-item.main-stat {
    font-size: 1.1rem;
    font-weight: bold;
    border-bottom: 1px solid #eee;
    margin-bottom: 0.75rem;
}

.stat-item.sub-stat {
    padding-left: 1.5rem;
    border-bottom: 1px solid #f9f9f9;
    font-size: 0.9rem;
}

.label {
    font-weight: 500;
}

.back-btn {
    padding: 0.85rem;
    background: #34495e;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.95rem;
    margin-top: 0.75rem;
    min-height: 44px;
    touch-action: manipulation;
}

.back-btn:hover {
    background: #2c3e50;
}

/* Mobile styles */
@media (max-width: 480px) {
    .stats-container {
        padding: 0 0.25rem;
        margin-top: 0.5rem;
    }

    .card {
        padding: 1rem;
        gap: 0.75rem;
        max-width: calc(100vw - 1rem);
    }

    h2 {
        font-size: 1.2rem;
    }

    h3 {
        font-size: 1rem;
        margin: 0.75rem 0 0.3rem 0;
    }

    h4 {
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }

    .stat-item {
        font-size: 0.85rem;
        padding: 0.35rem 0;
    }

    .stat-item.main-stat {
        font-size: 1rem;
    }

    .stat-item.sub-stat {
        padding-left: 1rem;
        font-size: 0.8rem;
    }

    .back-btn {
        padding: 0.7rem;
        font-size: 0.9rem;
    }
}

/* Very small screens */
@media (max-width: 360px) {
    .card {
        padding: 0.75rem;
    }

    h2 {
        font-size: 1.1rem;
    }

    .stat-item {
        font-size: 0.8rem;
    }

    .stat-item.sub-stat {
        font-size: 0.75rem;
    }
}
</style>
