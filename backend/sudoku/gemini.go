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

// CellConflict represents a conflict found in the puzzle
type CellConflict struct {
	Row     int    `json:"row"`
	Col     int    `json:"col"`
	Value   int    `json:"value"`
	Reason  string `json:"reason"`
}

// findConflicts checks a board for Sudoku rule violations and returns conflicting cells
func findConflicts(board Grid, size int) []CellConflict {
	var conflicts []CellConflict
	boxH, boxW := 3, 3
	if size == 6 {
		boxH, boxW = 2, 3
	}

	// Check rows
	for r := 0; r < size; r++ {
		seen := make(map[int]int) // value -> column
		for c := 0; c < size; c++ {
			val := board[r][c]
			if val == 0 {
				continue
			}
			if prevCol, exists := seen[val]; exists {
				conflicts = append(conflicts, CellConflict{
					Row: r, Col: c, Value: val,
					Reason: fmt.Sprintf("duplicate %d in row %d (also at col %d)", val, r, prevCol),
				})
			}
			seen[val] = c
		}
	}

	// Check columns
	for c := 0; c < size; c++ {
		seen := make(map[int]int) // value -> row
		for r := 0; r < size; r++ {
			val := board[r][c]
			if val == 0 {
				continue
			}
			if prevRow, exists := seen[val]; exists {
				conflicts = append(conflicts, CellConflict{
					Row: r, Col: c, Value: val,
					Reason: fmt.Sprintf("duplicate %d in column %d (also at row %d)", val, c, prevRow),
				})
			}
			seen[val] = r
		}
	}

	// Check boxes
	for boxR := 0; boxR < size/boxH; boxR++ {
		for boxC := 0; boxC < size/boxW; boxC++ {
			seen := make(map[int][2]int) // value -> [row, col]
			for r := boxR * boxH; r < (boxR+1)*boxH; r++ {
				for c := boxC * boxW; c < (boxC+1)*boxW; c++ {
					val := board[r][c]
					if val == 0 {
						continue
					}
					if prev, exists := seen[val]; exists {
						conflicts = append(conflicts, CellConflict{
							Row: r, Col: c, Value: val,
							Reason: fmt.Sprintf("duplicate %d in box (also at row %d, col %d)", val, prev[0], prev[1]),
						})
					}
					seen[val] = [2]int{r, c}
				}
			}
		}
	}

	return conflicts
}

// getCellImages fetches cell images from the extraction service
func getCellImages(imageBytes []byte, cells []CellConflict, size int) (map[string]string, error) {
	// Build cells array
	cellCoords := make([][2]int, len(cells))
	for i, c := range cells {
		cellCoords[i] = [2]int{c.Row, c.Col}
	}

	cellsJSON, _ := json.Marshal(cellCoords)

	// Create multipart request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("image", "puzzle.png")
	if err != nil {
		return nil, err
	}
	part.Write(imageBytes)

	writer.WriteField("size", fmt.Sprintf("%d", size))
	writer.WriteField("cells", string(cellsJSON))
	writer.Close()

	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("POST", LocalExtractionServiceURL+"/extract-cells", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cell extraction failed")
	}

	var result struct {
		Cells map[string]string `json:"cells"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	return result.Cells, nil
}

// extractClassicFromLocalService attempts to extract a classic sudoku puzzle using
// the local Python extraction service (OpenCV + EasyOCR based)
func extractClassicFromLocalService(imageBytes []byte) (Puzzle, error) {
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
	req, err := http.NewRequest("POST", LocalExtractionServiceURL+"/extract-classic", body)
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

	// Parse the response - classic sudoku only has board
	type ClassicExtractionResponse struct {
		Board Grid `json:"board"`
	}

	var localResp ClassicExtractionResponse
	if err := json.NewDecoder(resp.Body).Decode(&localResp); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to parse local extraction response: %v", err)
	}

	puzzle := Puzzle{
		Board:    localResp.Board,
		GameType: "standard",
	}

	if VerboseLogging {
		fmt.Printf("DEBUG: Classic local extraction successful\n")
	}

	return puzzle, nil
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

	// For killer sudoku, try local extraction service first (better cage detection)
	if gameType == "killer" && isLocalServiceAvailable() {
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
	}

	// For classic sudoku (and killer fallback), use Gemini
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
**CRITICAL INSTRUCTIONS FOR DIGIT EXTRACTION:**

1. Each cell may contain:
   - A LARGE, CENTERED digit (the actual placed number) - EXTRACT THIS
   - Small "pencil marks" (tiny numbers in corners/edges for candidates) - IGNORE THESE COMPLETELY
   - Nothing visible - use 0

2. **Size Rule**: Only extract digits that are LARGE and take up most of the cell's center area.
   - Large digits are typically 50-80% of the cell height
   - Small pencil marks are typically 10-25% of the cell height
   - If you see multiple small numbers scattered in a cell, that cell is EMPTY (use 0)

3. **Position Rule**: The digit must be CENTERED in the cell, not in corners or edges.

4. **When in doubt**: If you're unsure whether a number is a placed digit or a pencil mark, treat the cell as empty (0). It's better to miss a digit than to incorrectly include pencil marks.

**Output Format:**
Return ONLY a valid JSON object (no markdown, no code blocks):
{"board": [[row0], [row1], ..., [row8]]}

Each row is an array of 9 integers (0-9), where 0 means empty.`
	}

	mimeType := http.DetectContentType(imageBytes)
	// Default to jpeg if detection fails or is generic application/octet-stream,
	// though Gemini supports png, jpeg, webp, heic.
	if mimeType == "application/octet-stream" {
		mimeType = "image/jpeg"
	}

	// Use models that support image input
	model := "gemini-2.0-flash"
	if gameType == "killer" {
		model = "gemini-2.0-flash"
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

	// Get the text response (image models return text directly, no function calls)
	text := geminiResp.Candidates[0].Content.Parts[0].Text
	if text == "" {
		return emptyPuzzle, errors.New("no text response from gemini")
	}

	if VerboseLogging {
		fmt.Printf("DEBUG: Raw Gemini Response: %s\n", text)
	}

	puzzle, err := ParseGeminiResponse(text, gameType)
	if err != nil {
		return emptyPuzzle, err
	}

	// Validate the extracted puzzle and retry if there are conflicts
	size := len(puzzle.Board)
	if size == 0 {
		return puzzle, nil
	}

	conflicts := findConflicts(puzzle.Board, size)
	if len(conflicts) == 0 {
		// No conflicts, also try to solve to verify
		_, solvable := Solve(puzzle.Board, size)
		if solvable {
			return puzzle, nil
		}
		if VerboseLogging {
			fmt.Println("DEBUG: Puzzle has no conflicts but is unsolvable, attempting retry")
		}
	} else if VerboseLogging {
		fmt.Printf("DEBUG: Found %d conflicts in extracted puzzle\n", len(conflicts))
	}

	// If we have conflicts or puzzle is unsolvable, try to get cell images and retry
	if !isLocalServiceAvailable() {
		if VerboseLogging {
			fmt.Println("DEBUG: Local service unavailable for cell extraction, returning initial result")
		}
		return puzzle, nil
	}

	// Get unique conflicting cells (limit to first 6 to avoid too large request)
	uniqueCells := make(map[string]CellConflict)
	for _, c := range conflicts {
		key := fmt.Sprintf("%d,%d", c.Row, c.Col)
		uniqueCells[key] = c
	}

	cellList := make([]CellConflict, 0, len(uniqueCells))
	for _, c := range uniqueCells {
		cellList = append(cellList, c)
		if len(cellList) >= 6 {
			break
		}
	}

	if len(cellList) == 0 {
		// No specific conflicts but unsolvable - return as is
		return puzzle, nil
	}

	cellImages, err := getCellImages(imageBytes, cellList, size)
	if err != nil {
		if VerboseLogging {
			fmt.Printf("DEBUG: Failed to get cell images: %v\n", err)
		}
		return puzzle, nil
	}

	// Build a retry prompt with cell images
	retryPrompt := fmt.Sprintf(`The previous extraction of this Sudoku puzzle resulted in conflicts. Please re-examine these specific cells:

Current board state:
%s

Conflicts found:
`, formatBoard(puzzle.Board))

	for _, c := range cellList {
		retryPrompt += fmt.Sprintf("- Cell [%d,%d] has value %d: %s\n", c.Row, c.Col, c.Value, c.Reason)
	}

	retryPrompt += `
Below are zoomed images of the conflicting cells. Please look carefully at each cell image and determine the correct digit (1-9) or if it's empty (0).

IMPORTANT: Only extract the LARGE, CENTRAL digit. Ignore small pencil marks in corners.

Return a JSON object with corrections:
{
  "corrections": [
    {"row": <int>, "col": <int>, "value": <int>},
    ...
  ]
}
`

	// Build request with cell images
	retryParts := []GeminiPart{{Text: retryPrompt}}
	for coord, imgBase64 := range cellImages {
		retryParts = append(retryParts, GeminiPart{
			Text: fmt.Sprintf("Cell %s:", coord),
		})
		retryParts = append(retryParts, GeminiPart{
			InlineData: &GeminiInlineData{
				MimeType: "image/png",
				Data:     imgBase64,
			},
		})
	}

	retryReqBody := GeminiRequest{
		Contents: []GeminiContent{{Parts: retryParts}},
	}

	retryJSON, _ := json.Marshal(retryReqBody)
	retryResp, err := http.Post(url, "application/json", bytes.NewBuffer(retryJSON))
	if err != nil {
		return puzzle, nil
	}
	defer retryResp.Body.Close()

	if retryResp.StatusCode != http.StatusOK {
		return puzzle, nil
	}

	var retryGeminiResp GeminiResponse
	if err := json.NewDecoder(retryResp.Body).Decode(&retryGeminiResp); err != nil {
		return puzzle, nil
	}

	if len(retryGeminiResp.Candidates) == 0 || len(retryGeminiResp.Candidates[0].Content.Parts) == 0 {
		return puzzle, nil
	}

	retryText := retryGeminiResp.Candidates[0].Content.Parts[0].Text
	if VerboseLogging {
		fmt.Printf("DEBUG: Retry Gemini Response: %s\n", retryText)
	}

	// Parse corrections
	retryText = strings.TrimSpace(retryText)
	if strings.HasPrefix(retryText, "```json") {
		retryText = strings.TrimPrefix(retryText, "```json")
		retryText = strings.TrimSuffix(retryText, "```")
	} else if strings.HasPrefix(retryText, "```") {
		retryText = strings.TrimPrefix(retryText, "```")
		retryText = strings.TrimSuffix(retryText, "```")
	}
	retryText = strings.TrimSpace(retryText)

	var corrections struct {
		Corrections []struct {
			Row   int `json:"row"`
			Col   int `json:"col"`
			Value int `json:"value"`
		} `json:"corrections"`
	}

	if err := json.Unmarshal([]byte(retryText), &corrections); err != nil {
		if VerboseLogging {
			fmt.Printf("DEBUG: Failed to parse corrections: %v\n", err)
		}
		return puzzle, nil
	}

	// Apply corrections
	for _, corr := range corrections.Corrections {
		if corr.Row >= 0 && corr.Row < size && corr.Col >= 0 && corr.Col < size {
			if VerboseLogging {
				fmt.Printf("DEBUG: Applying correction [%d,%d]: %d -> %d\n",
					corr.Row, corr.Col, puzzle.Board[corr.Row][corr.Col], corr.Value)
			}
			puzzle.Board[corr.Row][corr.Col] = corr.Value
		}
	}

	return puzzle, nil
}

// formatBoard returns a string representation of the board for debugging
func formatBoard(board Grid) string {
	var sb strings.Builder
	for _, row := range board {
		sb.WriteString("[")
		for i, val := range row {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(fmt.Sprintf("%d", val))
		}
		sb.WriteString("]\n")
	}
	return sb.String()
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
