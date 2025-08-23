import pickle
import string
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value
    return landmarks.flatten()

# Load trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Webcam capture
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Prediction history for smoothing
prediction_history = deque(maxlen=20)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten(), hand_landmarks
    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, hand_landmarks = extract_landmarks(frame)
    if landmarks is not None:
        normalized = normalize_landmarks(landmarks)
        prediction = model.predict([normalized])[0]
        prediction_history.append(prediction)

        # Debug: print raw prediction and first 3 landmark triplets
        debug_sample = np.round(normalized[:9], 3)  # First 3 (x, y, z) triplets
        print(f"🧠 Raw prediction: {prediction} | Normalized sample: {debug_sample.tolist()}")

        # Smooth prediction using majority vote
        most_common = Counter(prediction_history).most_common(1)[0][0]

        cv2.putText(frame, f'Prediction: {most_common}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Debug: {debug_sample[:3]}...', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
