package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

type Stats struct {
	TotalSolved int `json:"totalSolved"`
	mu          sync.Mutex
}

var stats Stats

func main() {
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

		log.Printf("Generating puzzle: difficulty=%s, size=%d, type=%s", difficulty, size, gameType)

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

		log.Printf("Puzzle Completed: difficulty=%s, size=%d, type=%s", req.Difficulty, req.Size, req.GameType)

		stats.mu.Lock()
		stats.TotalSolved++
		stats.mu.Unlock()

		w.WriteHeader(http.StatusOK)
	})

	// API Endpoint: Stats
	mux.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		stats.mu.Lock()
		defer stats.mu.Unlock()

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
