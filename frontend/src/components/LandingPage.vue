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
        console.error('Upload error:', e)
        errorMsg.value = 'Failed to process image. Please try again.'
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
  <div class="flex justify-center items-center h-full w-full px-2 box-border">
    <div class="bg-white dark:bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg dark:shadow-none dark:border dark:border-gray-700 text-gray-800 dark:text-gray-100 flex flex-col gap-3 sm:gap-4 w-full max-w-md box-border">
      <h2 class="m-0 text-center text-gray-700 dark:text-gray-200 text-xl sm:text-2xl font-semibold">New Game Settings</h2>
        
      <div class="flex flex-col gap-1">
        <label class="font-bold text-sm">Game Type:</label>
        <div class="flex gap-1">
          <button
            class="flex-1 py-2 px-1 border rounded transition-all text-sm min-h-11 touch-manipulation"
            :class="gameType === 'standard' 
              ? 'bg-green-500 text-white border-green-500 font-bold' 
              : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-600'"
            @click="gameType = 'standard'"
          >
            Standard
          </button>
          <button
            class="flex-1 py-2 px-1 border rounded transition-all text-sm min-h-11 touch-manipulation"
            :class="gameType === 'killer' 
              ? 'bg-green-500 text-white border-green-500 font-bold' 
              : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-600'"
            @click="gameType = 'killer'"
          >
            Killer Sudoku
          </button>
        </div>
      </div>

      <div class="flex flex-col gap-1">
        <label class="font-bold text-sm">Grid Size:</label>
        <div class="flex gap-1">
          <button 
            class="flex-1 py-2 px-1 border rounded transition-all text-sm min-h-11 touch-manipulation"
            :class="size === 9 
              ? 'bg-green-500 text-white border-green-500 font-bold' 
              : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-600'"
            @click="size = 9"
          >
            Standard (9x9)
          </button>
          <button 
            class="flex-1 py-2 px-1 border rounded transition-all text-sm min-h-11 touch-manipulation"
            :class="size === 6 
              ? 'bg-green-500 text-white border-green-500 font-bold' 
              : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-600'"
            @click="size = 6"
          >
            Mini (6x6)
          </button>
        </div>
      </div>

      <div class="flex flex-col gap-1">
        <label class="font-bold text-sm">Difficulty:</label>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="diff in ['easy', 'medium', 'hard', 'extreme', 'insane']"
            :key="diff"
            class="py-2 px-1 border rounded transition-all text-xs sm:text-sm min-h-10 touch-manipulation"
            :class="[
              difficulty === diff 
                ? 'bg-green-500 text-white border-green-500 font-bold' 
                : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-600',
              diff === 'extreme' || diff === 'insane' ? 'flex-1 basis-[calc(50%-0.125rem)]' : 'flex-1 basis-[calc(33.33%-0.25rem)]'
            ]"
            @click="difficulty = diff"
          >
            {{ diff.charAt(0).toUpperCase() + diff.slice(1) }}
          </button>
        </div>
      </div>

      <div class="flex gap-3 mt-2">
        <button 
          class="flex-1 py-3 px-2 text-white text-base border-none rounded cursor-pointer min-h-12 touch-manipulation bg-gray-700 dark:bg-gray-600 hover:bg-gray-800 dark:hover:bg-gray-500"
          @click="startGame"
        >
          Start Game
        </button>
        <button
          class="flex-1 py-3 px-2 text-white text-base border-none rounded cursor-pointer min-h-12 touch-manipulation bg-orange-500 hover:bg-orange-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          @click="emit('create-custom', { size, gameType })"
        >
          Create Custom
        </button>
      </div>

      <div
        class="w-full mt-1"
        @dragover="handleDragOver"
        @dragleave="handleDragLeave"
        @drop="handleDrop"
      >
        <input
          type="file"
          ref="fileInput"
          accept="image/jpeg, image/png, image/webp"
          @change="handleFileUpload"
          class="hidden"
        />
        <div 
          class="w-full p-3 rounded cursor-pointer text-center transition-all box-border min-h-12 flex items-center justify-center border-2 border-dashed"
          :class="isDragging 
            ? 'bg-purple-200 dark:bg-purple-900 border-purple-700 dark:border-purple-400 border-solid' 
            : 'bg-purple-50 dark:bg-purple-900/30 border-purple-500 dark:border-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/50'"
          @click="triggerUpload"
        >
          <div v-if="isLoading" class="text-purple-600 dark:text-purple-300 font-medium text-sm">Processing...</div>
          <div v-else-if="isDragging" class="text-purple-600 dark:text-purple-300 font-medium text-sm">Drop image here</div>
          <div v-else class="flex items-center justify-center gap-2 text-purple-600 dark:text-purple-300 text-sm">
            <span class="text-lg font-bold">+</span>
            <span>Drop image here or click to upload</span>
          </div>
        </div>
        <div v-if="errorMsg" class="text-red-500 text-xs mt-1 text-center">{{ errorMsg }}</div>
      </div>

      <button 
        class="mt-2 py-3 px-4 bg-transparent text-gray-700 dark:text-gray-300 border border-gray-600 dark:border-gray-500 rounded cursor-pointer text-sm w-full min-h-11 touch-manipulation hover:bg-gray-100 dark:hover:bg-gray-700"
        @click="emit('view-stats')"
      >
        View Global Stats
      </button>
    </div>
  </div>
</template>
