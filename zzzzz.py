
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

MIN_AREA = 1200
ASPECT_MIN = 2.2
ASPECT_MAX = 4.5
COLOR_RATIO_THRESHOLD = 0.2
WHITE_RATIO_THRESHOLD = 0.35
BLACK_BAND_RATIO = 0.08
CANDIDATE_OVERLAP = 0.3
MIN_EXTENT = 0.5
MIN_SATURATION = 55

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


def detect_color(hsv_roi, color_hint):
    total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total_pixels == 0:
        return "UNKNOWN"

    if color_hint in ("RED", "GREEN", "BLUE"):
        mask = build_mask(hsv_roi, color_hint)
        ratio = cv2.countNonZero(mask) / total_pixels
        saturation_mean = float(np.mean(hsv_roi[:, :, 1]))
        if ratio >= COLOR_RATIO_THRESHOLD and saturation_mean >= MIN_SATURATION:
            return color_hint
        return "UNKNOWN"

    if color_hint == "WHITE":
        white_mask = build_mask(hsv_roi, "WHITE")
        white_ratio = cv2.countNonZero(white_mask) / total_pixels
        if white_ratio >= WHITE_RATIO_THRESHOLD and has_black_bands(hsv_roi):
            return "WHITE"
        return "UNKNOWN"

    return "UNKNOWN"


def boxes_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a[:4]
    bx, by, bw, bh = box_b[:4]
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    frame_area = frame.shape[0] * frame.shape[1]
    max_area = frame_area * 0.2
    candidates = []

    for color in ("RED", "GREEN", "BLUE", "WHITE"):
        mask = build_mask(hsv, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = h / w
            extent = area / float(w * h)
            if (
                ASPECT_MIN < aspect < ASPECT_MAX
                and h > w
                and extent >= MIN_EXTENT
            ):
                candidates.append((x, y, w, h, color))

    candidates = dedupe_boxes(candidates)

    detected_colors = []
    for x, y, w, h, color_hint in candidates:
        roi = hsv[y : y + h, x : x + w]
        color = detect_color(roi, color_hint)
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
