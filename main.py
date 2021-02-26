import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pymouse import PyMouse


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return int(resolution[0]), int(resolution[1])


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

NEXT_C = 0
NEXT_B = False

canvas = None

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    image_rows, image_cols = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))
    if not results.multi_hand_landmarks:
        continue
    annotated_image = cv2.flip(frame.copy(), 1)
    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = [mp_drawing._normalized_to_pixel_coordinates(l.x, l.y, image_cols, image_rows) for l in
                     hand_landmarks.landmark]
        if not all(landmarks):
            continue

        hand_w = max(landmarks, key=lambda x: x[0])[0] - min(landmarks, key=lambda x: x[0])[0]
        hand_h = max(landmarks, key=lambda x: x[1])[1] - min(landmarks, key=lambda x: x[1])[1]

        landmarks = np.array(landmarks)
        median_center = np.mean(landmarks[[1, 5, 9, 13, 17]], axis=0).astype(int)

        finger_x = landmarks[8][0]
        finger_y = landmarks[8][1]

        color = (255, 0, 0)
        if distance((finger_x / hand_w, finger_y / hand_h),
                    (median_center[0] / hand_w, median_center[1] / hand_h)) > 0.5:
            color = (0, 0, 255)
            NEXT_C = 0
            NEXT_B = False
            cv2.circle(canvas, (finger_x, finger_y), 5, (255, 255, 255), -1)
        else:
            NEXT_C += 1

        if NEXT_C > 5 and not NEXT_B:
            NEXT_B = True
            canvas = np.zeros_like(frame)
            print("Next!")

        cv2.circle(annotated_image, (median_center[0], median_center[1]), 5, (255, 0, 0), -1)
        cv2.circle(annotated_image, (finger_x, finger_y), 5, color, -1)

    cv2.imshow("Result", annotated_image)
    cv2.imshow("Mask", canvas)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
