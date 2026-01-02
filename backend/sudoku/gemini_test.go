package sudoku

import (
	"reflect"
	"testing"
)

func TestParseGeminiResponse(t *testing.T) {
	tests := []struct {
		name        string
		responseText string
		gameType    string
		wantPuzzle  Puzzle
		wantErr     bool
	}{
		{
			name: "Standard Sudoku",
			responseText: `{
				"board": [
					[5, 3, 0, 0, 7, 0, 0, 0, 0],
					[6, 0, 0, 1, 9, 5, 0, 0, 0],
					[0, 9, 8, 0, 0, 0, 0, 6, 0],
					[8, 0, 0, 0, 6, 0, 0, 0, 3],
					[4, 0, 0, 8, 0, 3, 0, 0, 1],
					[7, 0, 0, 0, 2, 0, 0, 0, 6],
					[0, 6, 0, 0, 0, 0, 2, 8, 0],
					[0, 0, 0, 4, 1, 9, 0, 0, 5],
					[0, 0, 0, 0, 8, 0, 0, 7, 9]
				]
			}`,
			gameType: "standard",
			wantPuzzle: Puzzle{
				Board: Grid{
					{5, 3, 0, 0, 7, 0, 0, 0, 0},
					{6, 0, 0, 1, 9, 5, 0, 0, 0},
					{0, 9, 8, 0, 0, 0, 0, 6, 0},
					{8, 0, 0, 0, 6, 0, 0, 0, 3},
					{4, 0, 0, 8, 0, 3, 0, 0, 1},
					{7, 0, 0, 0, 2, 0, 0, 0, 6},
					{0, 6, 0, 0, 0, 0, 2, 8, 0},
					{0, 0, 0, 4, 1, 9, 0, 0, 5},
					{0, 0, 0, 0, 8, 0, 0, 7, 9},
				},
				GameType: "standard",
			},
			wantErr: false,
		},
		{
			name: "Killer Sudoku with Cage Map",
			responseText: `{
				"board": [
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0]
				],
				"cage_map": [
					["A", "A", "B", "B", "C", "C", "D", "D", "D"],
					["E", "E", "F", "F", "G", "G", "H", "H", "H"],
					["I", "I", "J", "J", "K", "K", "L", "L", "L"],
					["M", "M", "N", "N", "O", "O", "P", "P", "P"],
					["Q", "Q", "R", "R", "S", "S", "T", "T", "T"],
					["U", "U", "V", "V", "W", "W", "X", "X", "X"],
					["Y", "Y", "Z", "Z", "1", "1", "2", "2", "2"],
					["3", "3", "4", "4", "5", "5", "6", "6", "6"],
					["7", "7", "8", "8", "9", "9", "0", "0", "0"]
				],
				"cage_sums": {
					"A": 10, "B": 20
				}
			}`,
			gameType: "killer",
			wantPuzzle: Puzzle{
				Board: make(Grid, 9),
				Cages: []Cage{
					{Sum: 10, Cells: []Point{{Row: 0, Col: 0}, {Row: 0, Col: 1}}},
					{Sum: 20, Cells: []Point{{Row: 0, Col: 2}, {Row: 0, Col: 3}}},
				},
				GameType: "killer",
			},
			wantErr: false,
		},
		{
			name: "Killer Sudoku with Legacy Cages",
			responseText: `{
				"board": [
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0]
				],
				"cages": [
					{
						"sum": 12,
						"cells": [{"row": 0, "col": 0}, {"row": 0, "col": 1}]
					},
					{
						"sum": 5,
						"cells": [{"row": 1, "col": 0}, {"row": 1, "col": 1}]
					}
				]
			}`,
			gameType: "killer",
			wantPuzzle: Puzzle{
				Board: make(Grid, 9), // 9x9 zero grid
				Cages: []Cage{
					{Sum: 12, Cells: []Point{{Row: 0, Col: 0}, {Row: 0, Col: 1}}},
					{Sum: 5, Cells: []Point{{Row: 1, Col: 0}, {Row: 1, Col: 1}}},
				},
				GameType: "killer",
			},
			wantErr: false,
		},
		{
			name: "Markdown Stripping",
			responseText: "```json\n" + `{
				"board": [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
			}` + "\n```",
			gameType: "standard",
			wantPuzzle: Puzzle{
				Board: make(Grid, 9),
				GameType: "standard",
			},
			wantErr: false,
		},
		{
			name: "Invalid JSON",
			responseText: `{ "board": [ ] }`, // Invalid size
			gameType: "standard",
			wantPuzzle: Puzzle{},
			wantErr: true,
		},
		{
			name: "Garbage Input",
			responseText: `This is not JSON`,
			gameType: "standard",
			wantPuzzle: Puzzle{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Helper to initialize empty grid for comparison if needed
			if len(tt.wantPuzzle.Board) == 9 && len(tt.wantPuzzle.Board[0]) == 0 {
				for i := range tt.wantPuzzle.Board {
					tt.wantPuzzle.Board[i] = make([]int, 9)
				}
			}
			if tt.wantPuzzle.Solution == nil {
				// ParseGeminiResponse doesn't return solution, so we expect nil or empty
			}

			got, err := ParseGeminiResponse(tt.responseText, tt.gameType)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseGeminiResponse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				// Compare GameType
				if got.GameType != tt.wantPuzzle.GameType {
					t.Errorf("ParseGeminiResponse() GameType = %v, want %v", got.GameType, tt.wantPuzzle.GameType)
				}
				// Compare Board
				if !reflect.DeepEqual(got.Board, tt.wantPuzzle.Board) {
					t.Errorf("ParseGeminiResponse() Board mismatch")
				}
				// Compare Cages
				if tt.gameType == "killer" {
					if len(got.Cages) != len(tt.wantPuzzle.Cages) {
						t.Errorf("ParseGeminiResponse() Cages length = %v, want %v", len(got.Cages), len(tt.wantPuzzle.Cages))
					} else {
						for i, cage := range got.Cages {
							if cage.Sum != tt.wantPuzzle.Cages[i].Sum {
								t.Errorf("ParseGeminiResponse() Cage[%d] Sum = %v, want %v", i, cage.Sum, tt.wantPuzzle.Cages[i].Sum)
							}
							// Point equality
							if !reflect.DeepEqual(cage.Cells, tt.wantPuzzle.Cages[i].Cells) {
								t.Errorf("ParseGeminiResponse() Cage[%d] Cells = %v, want %v", i, cage.Cells, tt.wantPuzzle.Cages[i].Cells)
							}
						}
					}
				}
			}
		})
	}
}
