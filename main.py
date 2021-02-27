import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import uuid
import pickle
from SpanishSpellingCorrector import word_chekcer, spanish_word_freq
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.backend import set_session
import pyautogui

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


def get_char(mask):
    mask = mask[:, :, 0]
    mask = cv2.dilate(mask, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(contour)
    return mask[y: y+h, x: x+w]


def predict_char(mask):
    mask = cv2.resize(mask, (100, 100))
    mask = mask / 255.
    mask = mask.reshape((1, 100, 100, 1)).astype('float32')
    res = model.predict(mask)[0]
    for k, v in dict_file.items():
        if v == np.argmax(res):
            return k


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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

NEXT_C = 0
NEXT_D = 0
NEXT_B = False
CORRECT = 0

BUFFER_SIZE = 7
BUFFER_IDX = 0
BUFFER_TOTAL_X = 0
BUFFER_TOTAL_Y = 0
buffer_x_coords = [0] * BUFFER_SIZE
buffer_y_coords = [0] * BUFFER_SIZE

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
model = load_model('model.h5')
a_file = open("model_loader.pkl", "rb")
dict_file = pickle.load(a_file)
a_file.close()

spanishWords = spanish_word_freq.SpanishWordFreq('./SpanishSpellingCorrector/10000_formas.TXT')
wordChecker = word_chekcer.WordChecker(spanishWords.words, spanishWords.totalFreq)

canvas = None
random_color = (128, 84, 244)

STRING = ""

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    image_rows, image_cols = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))
    if not results.multi_hand_landmarks:
        results.multi_hand_landmarks = []
        # continue
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

        BUFFER_TOTAL_X -= buffer_x_coords[BUFFER_IDX]
        BUFFER_TOTAL_Y -= buffer_y_coords[BUFFER_IDX]

        buffer_x_coords[BUFFER_IDX] = finger_x
        buffer_y_coords[BUFFER_IDX] = finger_y

        BUFFER_TOTAL_X += buffer_x_coords[BUFFER_IDX]
        BUFFER_TOTAL_Y += buffer_y_coords[BUFFER_IDX]

        BUFFER_IDX += 1

        if BUFFER_IDX >= BUFFER_SIZE:
            BUFFER_IDX = 0

        finger_x = BUFFER_TOTAL_X // BUFFER_SIZE
        finger_y = BUFFER_TOTAL_Y // BUFFER_SIZE

        color = (255, 0, 0)
        if distance((finger_x / hand_w, finger_y / hand_h),
                    (median_center[0] / hand_w, median_center[1] / hand_h)) > 0.5:
            color = (0, 0, 255)
            if NEXT_D < 4:
                NEXT_D += 1
            else:
                NEXT_C = 0
                NEXT_B = False
                cv2.circle(canvas, (finger_x, finger_y), 5, (255, 255, 255), -1)
        else:
            NEXT_C += 1

        if NEXT_C > 5 and not NEXT_B:
            NEXT_D = 0
            NEXT_B = True
            cut = get_char(canvas)
            char = predict_char(cut)
            if char == 'back' and len(STRING) > 0:
                STRING = STRING[:-1]
                pyautogui.press("backspace")
            elif char == 'space':
                if len(STRING) == 0:
                    continue
                CORRECT += 1
                if CORRECT < 2:
                    s = STRING.split(' ')[-1].lower()
                    try:
                        possible = wordChecker.candidates(s)[0][0]
                    except IndexError:
                        possible = s
                    if possible != s:
                        print(f"Maybe... \"{possible}\"? (draw space again to correct)")
                    else:
                        pyautogui.write(' ')
                        CORRECT = 0
                else:
                    CORRECT = 0
                    s = STRING.split(' ')
                    p = wordChecker.candidates(s[-1].lower())[0][0]
                    STRING = ' '.join(s[:-1] + [p]) + ' '
                    for _ in range(len(s[-1])):
                        pyautogui.press("backspace")
                    pyautogui.write(p)
                    pyautogui.write(' ')
            else:
                if CORRECT == 1:
                    STRING += ' '
                    pyautogui.write(' ')
                    CORRECT = 0
                if char != 'back':
                    STRING += char
                    pyautogui.write(char)
            # print(STRING)

            # plt.imshow(cut), plt.show()
            # cv2.imwrite(os.path.join('chars', 'y', str(uuid.uuid4()) + '.png'), cut)

            canvas = np.zeros_like(frame)
            random_color = list(np.random.choice(range(256), size=3))

        cv2.circle(annotated_image, (median_center[0], median_center[1]), 5, (255, 0, 0), -1)
        cv2.circle(annotated_image, (finger_x, finger_y), 5, color, -1)

    combined = cv2.addWeighted(np.full_like(annotated_image, random_color), 0.6, annotated_image, 0.4, 0)
    combined = cv2.add(np.uint8(combined * (canvas / 255.)), np.uint8(annotated_image * ((255 - canvas) / 255.)))
    cv2.imshow("Result", combined)
    # cv2.imshow("Mask", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
