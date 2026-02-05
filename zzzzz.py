
import cv2
import numpy as np
from collections import deque, Counter

cap = cv2.VideoCapture(1)

COLOR_RANGES = {
    "RED": [
        ((0, 80, 60), (10, 255, 255)),
        ((170, 80, 60), (180, 255, 255))
    ],
    "GREEN": [((35, 40, 40), (85, 255, 255))],
    "BLUE":  [((90, 50, 50), (130, 255, 255))],
    "WHITE": [((0, 0, 180), (180, 50, 255))]
}

MIN_AREA = 1800
ASPECT_MIN = 2.2
ASPECT_MAX = 4.0

last_box = None
lost_frames = 0
MAX_LOST = 15

COLOR_BUFFER_SIZE = 12
color_buffer = deque(maxlen=COLOR_BUFFER_SIZE)
stable_color = "UNKNOWN"

# 🔥 НОВОЕ
last_printed_color = None


def detect_color(hsv_roi):
    best_color = "UNKNOWN"
    best_count = 0

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv_roi, np.array(low), np.array(high))

        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color

    return best_color


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    best_candidate = None
    best_area = 0

    for ranges in COLOR_RANGES.values():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / w if w > 0 else 0

            if ASPECT_MIN < aspect < ASPECT_MAX:
                if area > best_area:
                    best_area = area
                    best_candidate = (x, y, w, h)

    if best_candidate is not None:
        last_box = best_candidate
        lost_frames = 0

        x, y, w, h = best_candidate
        roi = hsv[y:y+h, x:x+w]

        detected = detect_color(roi)
        if detected != "UNKNOWN":
            color_buffer.append(detected)

        if len(color_buffer) > 0:
            stable_color = Counter(color_buffer).most_common(1)[0][0]

        # 🖨️ ВЫВОД В КОНСОЛЬ
        if stable_color != last_printed_color and stable_color != "UNKNOWN":
            print(f"Detected cylinder color: {stable_color}")
            last_printed_color = stable_color

    else:
        lost_frames += 1

    if lost_frames > MAX_LOST:
        last_box = None
        color_buffer.clear()
        stable_color = "UNKNOWN"
        last_printed_color = None

    if last_box is not None:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(
            frame,
            stable_color,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Cylinder detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
