package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

type Stats struct {
	TotalSolved int                                  `json:"totalSolved"`
	Details     map[string]map[string]map[string]int `json:"details"`
}

var (
	stats   Stats
	statsMu sync.Mutex
)

const statsFile = "stats.json"

func loadStats() {
	file, err := os.Open(statsFile)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("Error opening stats file: %v", err)
		}
		return
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(&stats); err != nil {
		log.Printf("Error decoding stats file: %v", err)
	}
}

func saveStats() {
	file, err := os.Create(statsFile)
	if err != nil {
		log.Printf("Error creating stats file: %v", err)
		return
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(&stats); err != nil {
		log.Printf("Error encoding stats file: %v", err)
	}
}

func main() {
	log.Println("Starting Sudoku Server v1.1 (Stats V3)")
	loadStats()
	if stats.Details == nil {
		stats.Details = make(map[string]map[string]map[string]int)
	}

	mux := http.NewServeMux()

	// API Endpoint
	mux.HandleFunc("/api/puzzle", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		difficulty := r.URL.Query().Get("difficulty")
		sizeParam := r.URL.Query().Get("size")
		gameType := r.URL.Query().Get("gameType")
		size := 9
		if sizeParam == "6" {
			size = 6
		}

		cfIP := r.Header.Get("CF-Connecting-IP")
		sourceIP := r.RemoteAddr
		userAgent := r.UserAgent()
		log.Printf("Generating puzzle: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
			difficulty, size, gameType, cfIP, sourceIP, userAgent)

		var puzzle sudoku.Puzzle
		if gameType == "killer" {
			puzzle = sudoku.GenerateKiller(difficulty, size)
		} else {
			puzzle = sudoku.Generate(difficulty, size)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(puzzle)
	})

	// API Endpoint: Complete
	mux.HandleFunc("/api/complete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Difficulty string `json:"difficulty"`
			GameType   string `json:"gameType"`
			Size       int    `json:"size"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			// Just log error but don't fail hard if we can't parse details, we still want to count it?
			// Actually, let's just log "Unknown puzzle" if decode fails or proceed.
			log.Printf("Failed to decode completion request: %v", err)
		}

		cfIP := r.Header.Get("CF-Connecting-IP")
		sourceIP := r.RemoteAddr
		userAgent := r.UserAgent()
		log.Printf("Puzzle Completed: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
			req.Difficulty, req.Size, req.GameType, cfIP, sourceIP, userAgent)

		statsMu.Lock()
		stats.TotalSolved++

		if stats.Details == nil {
			stats.Details = make(map[string]map[string]map[string]int)
		}
		gType := req.GameType
		if gType == "" {
			gType = "standard"
		}
		if _, ok := stats.Details[gType]; !ok {
			stats.Details[gType] = make(map[string]map[string]int)
		}

		sizeStr := strconv.Itoa(req.Size)
		if _, ok := stats.Details[gType][sizeStr]; !ok {
			stats.Details[gType][sizeStr] = make(map[string]int)
		}

		stats.Details[gType][sizeStr][req.Difficulty]++

		saveStats()
		statsMu.Unlock()

		w.WriteHeader(http.StatusOK)
	})

	// API Endpoint: Stats
	mux.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		statsMu.Lock()
		defer statsMu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})

	// API Endpoint: Solve
	mux.HandleFunc("/api/solve", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Board [][]int `json:"board"`
			Size  int     `json:"size"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		solution, solvable := sudoku.Solve(req.Board, req.Size)
		if !solvable {
			// sudoku.Solve now returns false if count != 1.
			http.Error(w, "Invalid puzzle: Must have exactly one unique solution", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(struct {
			Solution [][]int `json:"solution"`
		}{
			Solution: solution,
		})
	})

	// API Endpoint: Upload Image
	mux.HandleFunc("/api/upload", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Limit upload size
		r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB

		file, _, err := r.FormFile("image")
		if err != nil {
			http.Error(w, "Invalid file", http.StatusBadRequest)
			return
		}
		defer file.Close()

		fileBytes, err := io.ReadAll(file)
		if err != nil {
			http.Error(w, "Error reading file", http.StatusInternalServerError)
			return
		}

		board, err := sudoku.ExtractSudokuFromImage(fileBytes)
		if err != nil {
			log.Printf("Gemini Error: %v", err)
			http.Error(w, "Failed to process image: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(struct {
			Board [][]int `json:"board"`
		}{
			Board: board,
		})
	})

	// Static File Serving
	// Check for "dist" (production) first, then fallback to "../frontend/dist" (local dev)
	frontendDir := "dist"
	if _, err := os.Stat(frontendDir); os.IsNotExist(err) {
		frontendDir = "../frontend/dist"
		if _, err := os.Stat(frontendDir); os.IsNotExist(err) {
			log.Printf("Warning: Frontend directory not found. Checked 'dist' and '../frontend/dist'.")
		}
	}

	log.Printf("Serving frontend from: %s", frontendDir)
	fs := http.FileServer(http.Dir(frontendDir))
	mux.Handle("/", fs)

	port := "8080"
	fmt.Printf("Server starting on port %s...\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Printf("Error starting server: %v\n", err)
		log.Fatal(err)
	}
	fmt.Println("Server stopped unexpectedly.")
}
