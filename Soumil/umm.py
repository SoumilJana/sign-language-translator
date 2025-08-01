import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Normalize function (same as before)
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value
    return landmarks.flatten()

# Extract landmarks from frame
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten(), hand_landmarks
    return None, None

# State for sentence logic
prev_prediction = ""
stable_start_time = time.time()
sentence = ""
prediction_history = deque(maxlen=20)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    landmarks, hand_landmarks = extract_landmarks(frame)

    if landmarks is not None:
        normalized = normalize_landmarks(landmarks)
        prediction = model.predict([normalized])[0]
        prediction_history.append(prediction)

        # Smooth prediction
        most_common = Counter(prediction_history).most_common(1)[0][0]

        # Check if stable for 4 seconds# Check if stable for 2 seconds
        if most_common == prev_prediction:
            if time.time() - stable_start_time >= 4:
                if most_common.lower() == "space":
                    if sentence == "" or sentence[-1] != " ":
                        sentence += " "
                        print("⬜ Space added")
                else:
                    if sentence == "" or sentence[-1] != most_common:
                        sentence += most_common.upper()
                        print(f"🟩 Added: {most_common} → Sentence: {sentence}")
                stable_start_time = time.time()
        else:
            prev_prediction = most_common
            stable_start_time = time.time()


        # Draw prediction & sentence
        # Draw white background rectangle
        cv2.rectangle(frame, (10, 20), (750, 120), (255, 255, 255), -1)

        # Draw black text over it
        cv2.putText(frame, f'Prediction: {most_common.upper()}', (20, 60),
            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

        cv2.putText(frame, f'Sentence: {sentence}', (20, 100),
            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)


        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Always draw the white background
    cv2.rectangle(frame, (10, 20), (750, 120), (255, 255, 255), -1)

    # First line: prediction or warning
    if hand_landmarks is not None:
        cv2.putText(frame, f'Prediction: {most_common.upper()}', (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, 'No hand detected', (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

    # Second line: sentence — always shown
    cv2.putText(frame, f'Sentence: {sentence}', (20, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)



    cv2.imshow("ASL Sentence Builder - Press Q to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Final Sentence:", sentence)

