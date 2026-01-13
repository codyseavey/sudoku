<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits(['start-game', 'create-custom', 'view-stats'])

const difficulty = ref('medium')
const size = ref(9)
const gameType = ref('standard')
const fileInput = ref<HTMLInputElement | null>(null)
const isLoading = ref(false)
const errorMsg = ref('')
const isDragging = ref(false)

const startGame = () => {
  emit('start-game', { difficulty: difficulty.value, size: size.value, gameType: gameType.value })
}

const triggerUpload = () => {
    errorMsg.value = ''
    fileInput.value?.click()
}

const processFile = async (file: File) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp']
    if (!validTypes.includes(file.type)) {
        errorMsg.value = 'Please upload a JPEG, PNG, or WebP image.'
        return
    }

    isLoading.value = true
    errorMsg.value = ''

    const formData = new FormData()
    formData.append('image', file)
    formData.append('gameType', gameType.value)

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        })

        if (!response.ok) {
            const errText = await response.text()
            throw new Error(errText || 'Upload failed')
        }

        const data = await response.json()
        if (data.board) {
            emit('create-custom', {
                size: 9,
                gameType: gameType.value,
                initialBoard: data.board,
                initialCages: data.cages
            })
            size.value = 9
        }
    } catch (e: any) {
        console.error("Upload error:", e)
        errorMsg.value = "Failed to process image. Please try again."
    } finally {
        isLoading.value = false
    }
}

const handleFileUpload = async (event: Event) => {
    const input = event.target as HTMLInputElement
    if (!input.files || input.files.length === 0) return

    const file = input.files[0]
    if (!file) return

    await processFile(file)

    if (fileInput.value) fileInput.value.value = ''
}

const handleDragOver = (event: DragEvent) => {
    event.preventDefault()
    isDragging.value = true
}

const handleDragLeave = (event: DragEvent) => {
    event.preventDefault()
    isDragging.value = false
}

const handleDrop = async (event: DragEvent) => {
    event.preventDefault()
    isDragging.value = false

    const files = event.dataTransfer?.files
    if (!files || files.length === 0) return

    const file = files[0]
    if (!file) return
    await processFile(file)
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
            <div class="options difficulty-options">
                <button
                    :class="{ active: difficulty === 'easy' }"
                    @click="difficulty = 'easy'"
                    title="Solvable with basic techniques only (singles)"
                >
                    Easy
                </button>
                <button
                    :class="{ active: difficulty === 'medium' }"
                    @click="difficulty = 'medium'"
                    title="May require pairs and pointing techniques"
                >
                    Medium
                </button>
                <button
                    :class="{ active: difficulty === 'hard' }"
                    @click="difficulty = 'hard'"
                    title="Requires intermediate techniques (X-Wing, Swordfish)"
                >
                    Hard
                </button>
                <button
                    :class="{ active: difficulty === 'extreme' }"
                    @click="difficulty = 'extreme'"
                    title="Requires advanced techniques (Y-Wing, Skyscraper)"
                >
                    Extreme
                </button>
                <button
                    :class="{ active: difficulty === 'insane' }"
                    @click="difficulty = 'insane'"
                    title="Requires multiple of the hardest techniques"
                >
                    Insane
                </button>
            </div>
        </div>

        <div class="buttons">
            <button class="start-btn" @click="startGame">Start Game</button>
            <button
                class="custom-btn"
                @click="emit('create-custom', { size, gameType })"
            >
                Create Custom
            </button>
        </div>

        <div
            class="upload-section"
            :class="{ 'drag-over': isDragging }"
            @dragover="handleDragOver"
            @dragleave="handleDragLeave"
            @drop="handleDrop"
        >
            <input
                type="file"
                ref="fileInput"
                accept="image/jpeg, image/png, image/webp"
                @change="handleFileUpload"
                style="display: none"
            />
            <div class="drop-zone" @click="triggerUpload">
                <div v-if="isLoading" class="upload-status">Processing...</div>
                <div v-else-if="isDragging" class="upload-status">Drop image here</div>
                <div v-else class="upload-prompt">
                    <span class="upload-icon">+</span>
                    <span>Drop image here or click to upload</span>
                </div>
            </div>
            <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>
        </div>

        <button class="stats-btn" @click="emit('view-stats')">View Global Stats</button>
    </div>
  </div>
</template>

<style scoped>
.landing-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    width: 100%;
    padding: 0 0.5rem;
    box-sizing: border-box;
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

.setting-group {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}

label {
    font-weight: bold;
    font-size: 0.85rem;
}

.options {
    display: flex;
    gap: 0.4rem;
}

.options button {
    flex: 1;
    padding: 0.5rem 0.25rem;
    border: 1px solid #ccc;
    background: #f8f9fa;
    color: #333;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
    font-size: 0.85rem;
    min-height: 44px;
    touch-action: manipulation;
}

.options button.active {
    background: #42b983;
    color: white;
    border-color: #42b983;
    font-weight: bold;
}

.difficulty-options {
    flex-wrap: wrap;
}

.difficulty-options button {
    flex: 0 1 calc(33.33% - 0.3rem);
    min-width: 0;
    font-size: 0.8rem;
    padding: 0.4rem 0.2rem;
}

.difficulty-options button:nth-child(4),
.difficulty-options button:nth-child(5) {
    flex: 0 1 calc(50% - 0.2rem);
    margin-top: 0.4rem;
}

.buttons {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.5rem;
}

.start-btn, .custom-btn {
    flex: 1;
    padding: 0.85rem 0.5rem;
    color: white;
    font-size: 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    min-height: 48px;
    touch-action: manipulation;
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

.upload-section {
    width: 100%;
    margin-top: 0.25rem;
}

.drop-zone {
    width: 100%;
    padding: 0.6rem;
    background: #f8f4fa;
    border: 2px dashed #8e44ad;
    border-radius: 4px;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s ease;
    box-sizing: border-box;
    min-height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.drop-zone:hover {
    background: #f0e6f4;
    border-color: #732d91;
}

.upload-section.drag-over .drop-zone {
    background: #e8d8f0;
    border-color: #5b2575;
    border-style: solid;
}

.upload-prompt {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: #8e44ad;
    font-size: 0.9rem;
}

.upload-icon {
    font-size: 1.1rem;
    font-weight: bold;
    line-height: 1;
}

.upload-status {
    color: #8e44ad;
    font-weight: 500;
    font-size: 0.9rem;
}

.error-msg {
    color: #e74c3c;
    font-size: 0.8rem;
    margin-top: 0.4rem;
    text-align: center;
}

.stats-btn {
    margin-top: 0.5rem;
    padding: 0.7rem;
    background: transparent;
    color: #34495e;
    border: 1px solid #34495e;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    width: 100%;
    min-height: 44px;
    touch-action: manipulation;
}

.stats-btn:hover {
    background: #f0f0f0;
}

/* Mobile styles */
@media (max-width: 480px) {
    .landing-container {
        padding: 0 0.25rem;
        align-items: flex-start;
        padding-top: 0.5rem;
    }

    .card {
        padding: 1rem;
        gap: 0.75rem;
        max-width: calc(100vw - 1rem);
    }

    h2 {
        font-size: 1.2rem;
    }

    .setting-group {
        gap: 0.3rem;
    }

    label {
        font-size: 0.8rem;
    }

    .options {
        gap: 0.3rem;
    }

    .options button {
        padding: 0.4rem 0.2rem;
        font-size: 0.75rem;
        min-height: 40px;
    }

    .difficulty-options button {
        font-size: 0.7rem;
        padding: 0.35rem 0.15rem;
    }

    .buttons {
        gap: 0.5rem;
    }

    .start-btn, .custom-btn {
        padding: 0.7rem 0.3rem;
        font-size: 0.9rem;
        min-height: 44px;
    }

    .drop-zone {
        padding: 0.5rem;
        min-height: 44px;
    }

    .upload-prompt {
        font-size: 0.8rem;
    }

    .stats-btn {
        padding: 0.6rem;
        font-size: 0.8rem;
        min-height: 40px;
    }
}

/* Very small screens */
@media (max-width: 360px) {
    .card {
        padding: 0.75rem;
        gap: 0.6rem;
    }

    h2 {
        font-size: 1.1rem;
    }

    .options button {
        font-size: 0.7rem;
        min-height: 36px;
    }

    .difficulty-options button {
        font-size: 0.65rem;
    }

    .start-btn, .custom-btn {
        font-size: 0.85rem;
        min-height: 40px;
    }
}
</style>
