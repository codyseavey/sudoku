package sudoku

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// LocalExtractionServiceURL is the URL of the Python extraction service
var LocalExtractionServiceURL = "http://127.0.0.1:5001"

var VerboseLogging bool

type GeminiRequest struct {
	Contents []GeminiContent `json:"contents"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *GeminiInlineData `json:"inline_data,omitempty"`
}

type GeminiInlineData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type GeminiResponse struct {
	Candidates []GeminiCandidate `json:"candidates"`
}

type GeminiCandidate struct {
	Content GeminiContent `json:"content"`
}

// extractFromLocalService attempts to extract a killer sudoku puzzle using
// the local Python extraction service (OpenCV + EasyOCR based)
func extractFromLocalService(imageBytes []byte, mimeType string) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// Create a multipart form request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add the image file
	part, err := writer.CreateFormFile("image", "puzzle.png")
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err := part.Write(imageBytes); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write image data: %v", err)
	}

	// Add size parameter
	if err := writer.WriteField("size", "9"); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write size field: %v", err)
	}

	if err := writer.Close(); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to close writer: %v", err)
	}

	// Make the request with a timeout
	client := &http.Client{Timeout: 60 * time.Second}
	req, err := http.NewRequest("POST", LocalExtractionServiceURL+"/extract", body)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("local extraction service unavailable: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return emptyPuzzle, fmt.Errorf("local extraction failed: %s", string(respBody))
	}

	// Parse the response
	type LocalExtractionResponse struct {
		Board    Grid              `json:"board"`
		CageMap  [][]string        `json:"cage_map"`
		CageSums map[string]int    `json:"cage_sums"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	var localResp LocalExtractionResponse
	if err := json.NewDecoder(resp.Body).Decode(&localResp); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to parse local extraction response: %v", err)
	}

	// Check if extraction was valid
	if localResp.Metadata != nil {
		if valid, ok := localResp.Metadata["valid"].(bool); ok && !valid {
			totalSum := 0
			if ts, ok := localResp.Metadata["total_sum"].(float64); ok {
				totalSum = int(ts)
			}
			if VerboseLogging {
				fmt.Printf("DEBUG: Local extraction invalid (sum=%d, expected=405)\n", totalSum)
			}
			// Continue anyway - the extraction might still be usable
		}
	}

	// Convert to Puzzle format
	puzzle := Puzzle{
		Board:    localResp.Board,
		GameType: "killer",
	}

	// Build cages from cage_map and cage_sums
	if len(localResp.CageMap) == 9 && len(localResp.CageSums) > 0 {
		cageCells := make(map[string][]Point)

		for r := 0; r < 9; r++ {
			if len(localResp.CageMap[r]) != 9 {
				return emptyPuzzle, fmt.Errorf("invalid cage_map row size at row %d", r)
			}
			for c := 0; c < 9; c++ {
				id := localResp.CageMap[r][c]
				cageCells[id] = append(cageCells[id], Point{Row: r, Col: c})
			}
		}

		for id, sum := range localResp.CageSums {
			if cells, ok := cageCells[id]; ok {
				sort.Slice(cells, func(i, j int) bool {
					if cells[i].Row != cells[j].Row {
						return cells[i].Row < cells[j].Row
					}
					return cells[i].Col < cells[j].Col
				})

				puzzle.Cages = append(puzzle.Cages, Cage{
					Sum:   sum,
					Cells: cells,
				})
			}
		}

		// Sort cages by the position of their first cell
		sort.Slice(puzzle.Cages, func(i, j int) bool {
			c1 := puzzle.Cages[i].Cells[0]
			c2 := puzzle.Cages[j].Cells[0]
			if c1.Row != c2.Row {
				return c1.Row < c2.Row
			}
			return c1.Col < c2.Col
		})
	}

	if VerboseLogging {
		totalSum := 0
		for _, cage := range puzzle.Cages {
			totalSum += cage.Sum
		}
		fmt.Printf("DEBUG: Local extraction successful. Cages=%d, TotalSum=%d\n", len(puzzle.Cages), totalSum)
	}

	return puzzle, nil
}

// isLocalServiceAvailable checks if the Python extraction service is running
func isLocalServiceAvailable() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(LocalExtractionServiceURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func ExtractSudokuFromImage(imageBytes []byte, gameType string) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// For killer sudoku, try local extraction service first
	if gameType == "killer" {
		if isLocalServiceAvailable() {
			if VerboseLogging {
				fmt.Println("DEBUG: Using local extraction service for killer sudoku")
			}
			mimeType := http.DetectContentType(imageBytes)
			puzzle, err := extractFromLocalService(imageBytes, mimeType)
			if err == nil {
				return puzzle, nil
			}
			if VerboseLogging {
				fmt.Printf("DEBUG: Local extraction failed, falling back to Gemini: %v\n", err)
			}
		} else if VerboseLogging {
			fmt.Println("DEBUG: Local extraction service not available, using Gemini")
		}
	}

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return emptyPuzzle, errors.New("GOOGLE_API_KEY not set")
	}

	encodedImage := base64.StdEncoding.EncodeToString(imageBytes)

	prompt := `Analyze the image of a Sudoku puzzle. Extract the puzzle state into a JSON object.`

	if gameType == "killer" {
		prompt += `
**Visual Guide for Killer Sudoku:**
- **Grid:** 9x9 grid.
- **Cages:** Defined by **dashed/dotted lines** surrounding groups of cells. These dashed lines are **hard boundaries**.
- **Cage Sums:** Small numbers usually located in the **top-left corner** of the first cell of a cage.
- **Placed Numbers:** Large, centered digits in cells. These go into the "board".
- **Empty Cells:** Cells with no large centered digit. Use 0.
- **Noise:** Ignore small pencil marks/candidates (multiple small numbers in a cell).

**Critical Rules for Cage Extraction:**
1. **Follow the Lines:** The dashed/dotted lines are the absolute reference. If there is a dashed line between two cells, they BELONG TO DIFFERENT CAGES (different IDs).
2. **Single-Cell Cages:** It is common to have cages that contain only 1 cell. This is especially true if the cell contains a placed number. If a cell is boxed in by dashed lines on all 4 sides, it is its own cage.
3. **Don't Guess:** Do not assume a cage "looks like" a 3-cell block. Look at the lines. If the lines isolate a cell, it is isolated.

**Output Requirements:**
Return ONLY a raw JSON object (no markdown, no code blocks) with this structure:
{
  "board": [[...], ...], // 9x9 array of integers (0-9)
  "cage_map": [[...], ...], // 9x9 array of strings. Each cell contains a unique ID (e.g. "a", "b", "c"...) identifying the cage it belongs to.
  "cage_sums": {
    "a": <int>,
    "b": <int>,
    ...
  } // Map where keys are the IDs from cage_map and values are the cage sums.
}

**Important:**
- Every cell (0,0) to (8,8) must have a cage ID in "cage_map".
- "cage_map" allows you to spatially define the cages, ensuring the shape is correct.
`
	} else {
		prompt += `
Rules:
1. Return ONLY a valid JSON object with a single field "board" containing a 9x9 array of integers.
2. Use 0 for empty cells.
3. CRITICAL: The puzzle contains "pencil marks" (small numbers in the corners/edges). IGNORE THESE.
4. Only extract the LARGE, CENTRAL digits that represent the placed numbers.
5. If a cell contains only small pencil marks, it is considered empty (0).
6. Do not include markdown formatting like ` + "`" + `` + "`" + `json` + "`" + `` + "`" + `.`
	}

	mimeType := http.DetectContentType(imageBytes)
	// Default to jpeg if detection fails or is generic application/octet-stream,
	// though Gemini supports png, jpeg, webp, heic.
	if mimeType == "application/octet-stream" {
		mimeType = "image/jpeg"
	}

	reqBody := GeminiRequest{
		Contents: []GeminiContent{
			{
				Parts: []GeminiPart{
					{Text: prompt},
					{
						InlineData: &GeminiInlineData{
							MimeType: mimeType,
							Data:     encodedImage,
						},
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return emptyPuzzle, err
	}

	model := "gemini-2.5-flash-image"
	if gameType == "killer" {
		// Use more advanced model for killer sudoku
		model = "gemini-3-pro-image-preview"

	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=", model) + apiKey
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return emptyPuzzle, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return emptyPuzzle, fmt.Errorf("gemini api error: %s", string(body))
	}

	var geminiResp GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		return emptyPuzzle, err
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return emptyPuzzle, errors.New("no content in gemini response")
	}

	text := geminiResp.Candidates[0].Content.Parts[0].Text

	if VerboseLogging {
		fmt.Printf("DEBUG: Raw Gemini Response: %s\n", text)
	}

	return ParseGeminiResponse(text, gameType)
}

func ParseGeminiResponse(text string, gameType string) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// Clean up markdown code blocks if present
	text = strings.TrimSpace(text)
	if strings.HasPrefix(text, "```json") {
		text = strings.TrimPrefix(text, "```json")
		text = strings.TrimSuffix(text, "```")
	} else if strings.HasPrefix(text, "```") {
		text = strings.TrimPrefix(text, "```")
		text = strings.TrimSuffix(text, "```")
	}
	text = strings.TrimSpace(text)

	// Temporary struct to handle the new format
	type GeminiRawResponse struct {
		Board    Grid           `json:"board"`
		CageMap  [][]string     `json:"cage_map"`
		CageSums map[string]int `json:"cage_sums"`
		Cages    []Cage         `json:"cages"` // Legacy support or fallthrough
	}

	var raw GeminiRawResponse
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		// Fallback: try parsing directly as Puzzle (old format might still be cached or returned?)
		// Actually, let's just error if it doesn't match expected structure or standard Puzzle structure
		// But since we changed the prompt, we expect the new structure.
		// However, standard games won't have cage_map.
		var simplePuzzle Puzzle
		if err2 := json.Unmarshal([]byte(text), &simplePuzzle); err2 == nil {
			simplePuzzle.GameType = gameType
			return simplePuzzle, nil
		}
		return emptyPuzzle, fmt.Errorf("failed to parse board json: %v. Text was: %s", err, text)
	}

	if len(raw.Board) != 9 {
		// Try standard puzzle parse
		var simplePuzzle Puzzle
		if err := json.Unmarshal([]byte(text), &simplePuzzle); err == nil && len(simplePuzzle.Board) == 9 {
			simplePuzzle.GameType = gameType
			return simplePuzzle, nil
		}
		return emptyPuzzle, fmt.Errorf("invalid board size: %d", len(raw.Board))
	}

	puzzle := Puzzle{
		Board:    raw.Board,
		GameType: gameType,
	}

	// If we have a cage map, reconstruct cages
	if len(raw.CageMap) == 9 && len(raw.CageSums) > 0 {
		cageCells := make(map[string][]Point)

		for r := 0; r < 9; r++ {
			if len(raw.CageMap[r]) != 9 {
				return emptyPuzzle, fmt.Errorf("invalid cage_map row size at row %d", r)
			}
			for c := 0; c < 9; c++ {
				id := raw.CageMap[r][c]
				cageCells[id] = append(cageCells[id], Point{Row: r, Col: c})
			}
		}

		for id, sum := range raw.CageSums {
			if cells, ok := cageCells[id]; ok {
				// Sort cells within the cage to be deterministic (Row then Col)
				sort.Slice(cells, func(i, j int) bool {
					if cells[i].Row != cells[j].Row {
						return cells[i].Row < cells[j].Row
					}
					return cells[i].Col < cells[j].Col
				})

				puzzle.Cages = append(puzzle.Cages, Cage{
					Sum:   sum,
					Cells: cells,
				})
			}
		}

		// Sort cages by the position of their first cell
		sort.Slice(puzzle.Cages, func(i, j int) bool {
			c1 := puzzle.Cages[i].Cells[0]
			c2 := puzzle.Cages[j].Cells[0]
			if c1.Row != c2.Row {
				return c1.Row < c2.Row
			}
			return c1.Col < c2.Col
		})
	} else if len(raw.Cages) > 0 {
		// Fallback if model returned old format despite instructions (unlikely but safe)
		puzzle.Cages = raw.Cages
	}

	return puzzle, nil
}
