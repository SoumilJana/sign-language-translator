import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import pyttsx3  # Text-to-Speech

# Load model
model_dict = pickle.load(open('model.p', 'rb'))  # Make sure this file exists
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value
    return landmarks.flatten()

# Extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]), hand_landmarks
    return None, None

# State
prev_prediction = ""
stable_start_time = time.time()
sentence = ""
prediction_history = deque(maxlen=20)
prev_frame_time = time.time()

# Speak function with reinit
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 110)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ TTS Error: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    landmarks_raw, hand_landmarks = extract_landmarks(frame)

    if landmarks_raw is not None:
        normalized = normalize_landmarks(landmarks_raw.flatten())
        prediction = model.predict([normalized])[0]
        prediction_history.append(prediction)
        most_common = Counter(prediction_history).most_common(1)[0][0]

        # Stable prediction logic
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

    # Draw UI
    cv2.rectangle(frame, (10, 20), (750, 120), (255, 255, 255), -1)

    if hand_landmarks is not None:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Prediction: {most_common.upper()}', (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
    else:
        cv2.putText(frame, 'No hand detected', (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

    # Sentence display
    cv2.putText(frame, f'Sentence: {sentence}', (20, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display window
    cv2.imshow("ASL Sentence Builder - Press Enter to Finalize | Q/ESC to Quit", frame)

    # Key detection
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:  # Quit
        break
    elif key == 13:  # Enter key (↵)
        if sentence.strip():
            print("\n🟢 Enter Pressed! Final Sentence:", sentence)
            speak(sentence)
            with open("output_sentence.txt", "w") as f:
                f.write(sentence)
            print("💾 Sentence saved to: output_sentence.txt")
            sentence = ""
            prediction_history.clear()
            stable_start_time = time.time()
            prev_prediction = ""
    elif key == 8:  # Backspace/Delete key
        if sentence:
            sentence = sentence[:-1]
            print(f"⬅️ Deleted last letter → Sentence: {sentence}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
