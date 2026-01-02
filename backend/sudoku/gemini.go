package sudoku

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type GeminiRequest struct {
	Contents []GeminiContent `json:"contents"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text       string           `json:"text,omitempty"`
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

func ExtractSudokuFromImage(imageBytes []byte) ([][]int, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, errors.New("GOOGLE_API_KEY not set")
	}

	encodedImage := base64.StdEncoding.EncodeToString(imageBytes)

	prompt := `Analyze the image of a Sudoku puzzle. Extract the 9x9 grid state into a JSON object.
Rules:
1. Return ONLY a valid JSON object with a single field "board" containing a 9x9 array of integers.
2. Use 0 for empty cells.
3. CRITICAL: The puzzle contains "pencil marks" (small numbers in the corners/edges). IGNORE THESE.
4. Only extract the LARGE, CENTRAL digits that represent the placed numbers.
5. If a cell contains only small pencil marks, it is considered empty (0).
6. Do not include markdown formatting like ` + "`" + `` + "`" + `json` + "`" + `` + "`" + `.`

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
		return nil, err
	}

	// Use gemini-2.0-flash as 1.5-flash caused 404s
	url := "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + apiKey
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini api error: %s", string(body))
	}

	var geminiResp GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		return nil, err
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("no content in gemini response")
	}

	text := geminiResp.Candidates[0].Content.Parts[0].Text

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


	var result struct {
		Board [][]int `json:"board"`
	}

	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse board json: %v. Text was: %s", err, text)
	}

    if len(result.Board) != 9 {
        return nil, fmt.Errorf("invalid board size: %d", len(result.Board))
    }

	return result.Board, nil
}
