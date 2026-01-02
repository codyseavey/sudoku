
import cv2
import numpy as np
import os
import json
import networkx as nx
import shutil
import sys

# --- Utilities ---

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def extract_grid_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzle_cnt = approx
            break
    if puzzle_cnt is None: return None
    warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))
    warped = cv2.resize(warped, (900, 900), interpolation=cv2.INTER_AREA)
    return warped

def get_cell(grid_img, r, c):
    h, w = grid_img.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    return grid_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]

def calculate_boundary_features(strip, orientation):
    strip_blur = cv2.GaussianBlur(strip, (3,3), 0)
    thresh = cv2.adaptiveThreshold(strip_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    if orientation == 'h':
        profile = np.mean(thresh, axis=0)
    else:
        profile = np.mean(thresh, axis=1)

    mean_val = np.mean(profile)
    centered = profile - mean_val
    zc = ((centered[:-1] * centered[1:]) < 0).sum()
    return zc

def learn_boundary_threshold(warped, json_data):
    cage_map = json_data["cage_map"]
    cell_h = 900 // 9
    cell_w = 900 // 9

    same_scores = []
    diff_scores = []

    # Horizontal
    for r in range(8):
        for c in range(9):
            y = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            strip = warped[y-3:y+3, x1+5:x2-5]
            score = calculate_boundary_features(strip, 'h')

            if cage_map[r][c] == cage_map[r+1][c]:
                same_scores.append(score)
            else:
                diff_scores.append(score)

    # Vertical
    for r in range(9):
        for c in range(8):
            x = (c + 1) * cell_w
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            strip = warped[y1+5:y2-5, x-3:x+3]
            score = calculate_boundary_features(strip, 'v')

            if cage_map[r][c] == cage_map[r][c+1]:
                same_scores.append(score)
            else:
                diff_scores.append(score)

    mean_same = np.mean(same_scores) if same_scores else 0
    mean_diff = np.mean(diff_scores) if diff_scores else 0

    threshold = (mean_same + mean_diff) / 2
    print(f"Learned Boundary Threshold: {threshold:.2f} (Same Mean: {mean_same:.2f}, Diff Mean: {mean_diff:.2f})")
    return threshold

def extract_templates_and_learn(image_path, json_path, template_dir="templates"):
    if os.path.exists(template_dir):
        shutil.rmtree(template_dir)
    os.makedirs(f"{template_dir}/digits", exist_ok=True)
    os.makedirs(f"{template_dir}/small_digits", exist_ok=True)

    warped = extract_grid_image(image_path)
    if warped is None:
        print(f"Failed to extract grid from {image_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    boundary_threshold = learn_boundary_threshold(warped, data)

    board = data["board"]
    cage_map = data["cage_map"]
    cage_sums = data["cage_sums"]

    # 1. Big Digits
    for r in range(9):
        for c in range(9):
            digit = board[r][c]
            if digit != 0:
                cell = get_cell(warped, r, c)
                h, w = cell.shape
                roi = cell[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
                roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c_max = max(cnts, key=cv2.contourArea)
                    x, y, w_c, h_c = cv2.boundingRect(c_max)
                    digit_img = roi_thresh[y:y+h_c, x:x+w_c]
                    cv2.imwrite(f"{template_dir}/digits/{digit}_{r}_{c}.png", digit_img)

    # 2. Small Digits
    visited_cages = set()
    for r in range(9):
        for c in range(9):
            cage_id = cage_map[r][c]
            if cage_id not in visited_cages:
                visited_cages.add(cage_id)
                sum_val = cage_sums[cage_id]
                sum_str = str(sum_val)

                cell = get_cell(warped, r, c)
                h, w = cell.shape
                roi = cell[2:int(h*0.4), 2:int(w*0.4)]
                roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_h = h * 0.05
                max_h = h * 0.35
                valid_cnts = []
                for cnt in cnts:
                    x, y, w_c, h_c = cv2.boundingRect(cnt)
                    if h_c > min_h and h_c < max_h and w_c > 2:
                        valid_cnts.append(cnt)

                valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)
                candidates = valid_cnts[:len(sum_str)]
                candidates = sorted(candidates, key=lambda c: cv2.boundingRect(c)[0])

                if len(candidates) == len(sum_str):
                    for i, cnt in enumerate(candidates):
                        x, y, w_c, h_c = cv2.boundingRect(cnt)
                        digit_img = roi_thresh[y:y+h_c, x:x+w_c]
                        digit_char = sum_str[i]
                        cv2.imwrite(f"{template_dir}/small_digits/{digit_char}_{cage_id}_{i}.png", digit_img)

    print("Templates extracted.")
    return boundary_threshold

def load_templates(template_dir):
    templates = {}
    if not os.path.exists(template_dir):
        return templates
    for f in os.listdir(template_dir):
        if f.endswith(".png") and not f.startswith("FAIL"):
            digit = f.split("_")[0]
            if digit not in templates:
                templates[digit] = []
            img = cv2.imread(os.path.join(template_dir, f), cv2.IMREAD_GRAYSCALE)
            templates[digit].append(img)
    return templates

def match_template(roi, templates, threshold=0.5, debug=False):
    # Fixed size matching
    if roi.shape[0] == 0 or roi.shape[1] == 0: return None
    roi_resized = cv2.resize(roi, (20, 30))

    best_score = -1
    best_digit = None

    for digit, templ_list in templates.items():
        for templ in templ_list:
            t_resized = cv2.resize(templ, (20, 30))
            score = cv2.matchTemplate(roi_resized, t_resized, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_digit = digit

    if debug:
        print(f"    Best match: {best_digit} (score: {best_score:.2f})")

    if best_score > threshold:
        return best_digit
    return None

def process_image(image_path, big_templates, small_templates, boundary_threshold, debug=False):
    warped = extract_grid_image(image_path)
    if warped is None:
        print(f"Failed to extract grid for {image_path}")
        return None

    board = [[0]*9 for _ in range(9)]

    # 1. Recognize Givens
    for r in range(9):
        for c in range(9):
            cell = get_cell(warped, r, c)
            h, w = cell.shape
            roi = cell[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            if cv2.countNonZero(roi_thresh) < 50:
                continue

            cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            c_max = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c_max) < 100: continue

            x, y, w_c, h_c = cv2.boundingRect(c_max)
            digit_img = roi_thresh[y:y+h_c, x:x+w_c]

            digit = match_template(digit_img, big_templates, threshold=0.35)
            if digit:
                board[r][c] = int(digit)

    # 2. Determine Cages
    G = nx.Graph()
    for r in range(9):
        for c in range(9):
            G.add_node((r,c))

    cell_h = 900 // 9
    cell_w = 900 // 9

    for r in range(8):
        for c in range(9):
            y = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            strip = warped[y-3:y+3, x1+5:x2-5]
            zc = calculate_boundary_features(strip, 'h')
            is_boundary = zc > boundary_threshold
            if not is_boundary:
                G.add_edge((r,c), (r+1,c))

    for r in range(9):
        for c in range(8):
            x = (c + 1) * cell_w
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            strip = warped[y1+5:y2-5, x-3:x+3]
            zc = calculate_boundary_features(strip, 'v')
            is_boundary = zc > boundary_threshold
            if not is_boundary:
                G.add_edge((r,c), (r,c+1))

    components = list(nx.connected_components(G))
    cage_map = [[None]*9 for _ in range(9)]
    cage_sums = {}

    cage_ids = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, comp in enumerate(components):
        if idx >= len(cage_ids): cid = f"c{idx}"
        else: cid = cage_ids[idx]

        sorted_cells = sorted(list(comp), key=lambda p: (p[0], p[1]))
        top_left = sorted_cells[0]
        r, c = top_left

        cell = get_cell(warped, r, c)
        h, w = cell.shape
        roi = cell[2:int(h*0.4), 2:int(w*0.4)]
        roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_h = h * 0.05
        max_h = h * 0.35
        valid_cnts = []
        for cnt in cnts:
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if h_c > min_h and h_c < max_h and w_c > 2:
                valid_cnts.append(cnt)

        valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)
        # No slicing limit, rely on threshold
        candidates = valid_cnts
        candidates = sorted(candidates, key=lambda c: cv2.boundingRect(c)[0])

        sum_str = ""
        for cnt in candidates:
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            digit_img = roi_thresh[y_c:y_c+h_c, x_c:x_c+w_c]

            # Increased threshold to 0.6 to filter noise
            d = match_template(digit_img, small_templates, threshold=0.6, debug=(debug and "1.png" in image_path))
            if d:
                sum_str += d

        if not sum_str:
            sum_val = 0
            if debug and "1.png" in image_path:
                print(f"  Cage {cid} at {r},{c}: No digits found.")
        else:
            try:
                sum_val = int(sum_str)
            except:
                sum_val = 0

        cage_sums[cid] = sum_val
        if debug and "1.png" in image_path:
            print(f"  Cage {cid} at {r},{c}: Sum {sum_val} (raw: {sum_str})")

        for (r_c, c_c) in comp:
            cage_map[r_c][c_c] = cid

    return {"board": board, "cage_map": cage_map, "cage_sums": cage_sums}

def main():
    if len(sys.argv) < 2:
        data_dir = "backend/test_data/killer_sudoku"
        learn_img = os.path.join(data_dir, "1.png")
        learn_json = os.path.join(data_dir, "1.json")
    else:
        data_dir = sys.argv[1]
        learn_img = sys.argv[2]
        learn_json = sys.argv[3]

    if not os.path.exists(learn_img) or not os.path.exists(learn_json):
        print("Learning data not found. Cannot proceed.")
        return

    # 1. Extract templates and learn threshold from learning data
    print(f"Learning from {learn_img}...")
    boundary_threshold = extract_templates_and_learn(learn_img, learn_json)
    if boundary_threshold is None:
        boundary_threshold = 5.0 # Fallback

    big_templates = load_templates("templates/digits")
    small_templates = load_templates("templates/small_digits")
    if '7' not in small_templates and '7' in big_templates:
        small_templates['7'] = big_templates['7']

    # 2. Process all images
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])
    for f in files:
        if f == "1.png":
            print(f"Processing {f} (Reference)...")
            res = process_image(os.path.join(data_dir, f), big_templates, small_templates, boundary_threshold, debug=True)
            if res:
                total_sum = sum(res["cage_sums"].values())
                print(f"  Total Cage Sum: {total_sum}")
        elif f == os.path.basename(learn_img):
            continue
        else:
            print(f"Processing {f}...")
            res = process_image(os.path.join(data_dir, f), big_templates, small_templates, boundary_threshold)
            if res:
                total_sum = sum(res["cage_sums"].values())
                print(f"  Total Cage Sum: {total_sum}")
                if total_sum != 405:
                    print(f"  WARNING: Sum is {total_sum}, expected 405. Skipping file generation.")
                else:
                    out_name = f.replace(".png", ".json")
                    out_path = os.path.join(data_dir, out_name)
                    with open(out_path, "w") as jf:
                        json.dump(res, jf, indent=2)
                    print(f"  Generated {out_path}")

if __name__ == "__main__":
    main()
