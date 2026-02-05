
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

COLOR_RANGES = {
    "RED": [
        ((0, 60, 40), (10, 255, 255)),
        ((170, 60, 40), (180, 255, 255)),
    ],
    "GREEN": [((35, 40, 40), (85, 255, 255))],
    "BLUE": [((90, 40, 40), (130, 255, 255))],
    "WHITE": [((0, 0, 120), (180, 70, 255))],
    "BLACK": [((0, 0, 0), (180, 70, 60))],
}

COLOR_NAMES_RU = {
    "RED": "красный",
    "GREEN": "зелёный",
    "BLUE": "синий",
    "WHITE": "белый",
}

DRAW_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "WHITE": (255, 255, 255),
    "UNKNOWN": (0, 0, 255),
}

MIN_AREA = 600
ASPECT_MIN = 1.8
ASPECT_MAX = 5.5
EXPECTED_ASPECT = 3.0
ASPECT_TOLERANCE = 1.6
COLOR_RATIO_THRESHOLD = 0.12
WHITE_RATIO_THRESHOLD = 0.2
BLACK_BAND_RATIO = 0.05
CANDIDATE_OVERLAP = 0.4
BLACK_BAND_MIN_AREA = 80

COLOR_TTL = 15
last_seen = {}

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def build_mask(hsv_roi, color):
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for low, high in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv_roi, np.array(low), np.array(high))
    return mask


def has_black_bands(hsv_roi):
    mask_black = build_mask(hsv_roi, "BLACK")
    height = hsv_roi.shape[0]
    band = max(1, int(height * 0.15))
    top_band = mask_black[:band, :]
    bottom_band = mask_black[-band:, :]
    top_ratio = cv2.countNonZero(top_band) / top_band.size
    bottom_ratio = cv2.countNonZero(bottom_band) / bottom_band.size
    return top_ratio > BLACK_BAND_RATIO and bottom_ratio > BLACK_BAND_RATIO


def detect_color(hsv_roi):
    total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total_pixels == 0:
        return "UNKNOWN"

    scores = {}
    for color in ("RED", "GREEN", "BLUE"):
        mask = build_mask(hsv_roi, color)
        scores[color] = cv2.countNonZero(mask) / total_pixels

    white_mask = build_mask(hsv_roi, "WHITE")
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    best_color = max(scores, key=scores.get)
    if scores[best_color] >= COLOR_RATIO_THRESHOLD:
        return best_color

    saturation_mean = float(np.mean(hsv_roi[:, :, 1]))
    if white_ratio >= WHITE_RATIO_THRESHOLD and (
        has_black_bands(hsv_roi) or saturation_mean < 40
    ):
        return "WHITE"

    return "UNKNOWN"


def boxes_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def dedupe_boxes(boxes, overlap_threshold=CANDIDATE_OVERLAP):
    result = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if any(boxes_overlap(box, kept) > overlap_threshold for kept in result):
            continue
        result.append(box)
    return result


def clamp_box(x, y, w, h, frame_width, frame_height):
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return x, y, w, h


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    median_val = float(np.median(gray))
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    max_area = frame_area * 0.2
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0:
            continue
        aspect = h / w
        if ASPECT_MIN < aspect < ASPECT_MAX and h > w:
            candidates.append((x, y, w, h))

    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color in ("RED", "GREEN", "BLUE"):
        color_mask |= build_mask(hsv, color)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    color_contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in color_contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0:
            continue
        aspect = h / w
        if ASPECT_MIN < aspect < ASPECT_MAX and h > w:
            candidates.append((x, y, w, h))

    black_mask = build_mask(hsv, "BLACK")
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black_contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in black_contours:
        area = cv2.contourArea(cnt)
        if area < BLACK_BAND_MIN_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            continue
        if w < h * 2:
            continue
        band_center_x = x + w // 2
        band_center_y = y + h // 2
        box_w = w
        box_h = int(w * EXPECTED_ASPECT)
        if box_h < h * 2:
            box_h = int(h * EXPECTED_ASPECT * 2)
        box_x = int(band_center_x - box_w // 2)
        box_y = int(band_center_y - box_h // 2)
        box_x, box_y, box_w, box_h = clamp_box(
            box_x, box_y, box_w, box_h, frame.shape[1], frame.shape[0]
        )
        aspect = box_h / box_w
        if EXPECTED_ASPECT - ASPECT_TOLERANCE < aspect < EXPECTED_ASPECT + ASPECT_TOLERANCE:
            candidates.append((box_x, box_y, box_w, box_h))

    candidates = dedupe_boxes(candidates)

    detected_colors = []
    for x, y, w, h in candidates:
        roi = hsv[y : y + h, x : x + w]
        color = detect_color(roi)
        if color == "UNKNOWN":
            continue

        detected_colors.append(color)
        draw_color = DRAW_COLORS.get(color, DRAW_COLORS["UNKNOWN"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        cv2.putText(
            frame,
            color,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            draw_color,
            2,
        )

    for color in detected_colors:
        if color not in last_seen:
            print(f"обнаружен цилиндр цвета {COLOR_NAMES_RU[color]}")
        last_seen[color] = COLOR_TTL

    for color in list(last_seen.keys()):
        last_seen[color] -= 1
        if last_seen[color] <= 0:
            last_seen.pop(color, None)

    cv2.imshow("Cylinder detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
