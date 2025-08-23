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
last_added = ""   
last_added_time = 0  
cooldown = 0.5    
stable_start_time = time.time()
sentence = ""
prediction_history = deque(maxlen=10)  # smoothing buffer
prev_frame_time = time.time()

# Speak function
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
    h, w, _ = frame.shape
    landmarks_raw, hand_landmarks = extract_landmarks(frame)

    if landmarks_raw is not None:
        normalized = normalize_landmarks(landmarks_raw.flatten())
        prediction = model.predict([normalized])[0]

        # Confidence calculation
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([normalized])[0]
            confidence = np.max(probs) * 100
        else:
            confidence = 100.0  

        # === Smooth prediction ===
        prediction_history.append(prediction)
        smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]

        # Stable prediction logic (for adding letters to sentence)
        if smoothed_prediction == prev_prediction:
            if time.time() - stable_start_time >= 1:  # 1 second stability
                current_time = time.time()

                if smoothed_prediction.lower() == "space":
                    if last_added != " " or (current_time - last_added_time) > cooldown:
                        sentence += " "
                        last_added = " "
                        last_added_time = current_time
                        print("⬜ Space added")
                else:
                    letter = smoothed_prediction.upper()
                    if letter != last_added or (current_time - last_added_time) > cooldown:
                        sentence += letter
                        last_added = letter
                        last_added_time = current_time
                        print(f"🟩 Added: {letter} → Sentence: {sentence}")

                stable_start_time = time.time()
        else:
            prev_prediction = smoothed_prediction
            stable_start_time = time.time()

        # === Show smoothed prediction beside hand ===
        x_min = int(min([lm[0] for lm in landmarks_raw]) * w)
        y_min = int(min([lm[1] for lm in landmarks_raw]) * h)

        cv2.putText(frame, f"{smoothed_prediction.upper()} {confidence:.1f}%",
                    (x_min, max(30, y_min - 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

    if hand_landmarks is not None:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, 'No hand detected', (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    # === Sentence display with black strip at bottom ===
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, f'Sentence: {sentence}', (20, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # FPS (top-right)
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Display window in fullscreen
    cv2.namedWindow("ASL Sentence Builder - Press Enter to Finalize | Q/ESC to Quit", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("ASL Sentence Builder - Press Enter to Finalize | Q/ESC to Quit",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            last_added = ""
            last_added_time = 0
    elif key == 8:  # Backspace/Delete key
        if sentence:
            sentence = sentence[:-1]
            last_added = sentence[-1] if sentence else ""
            last_added_time = time.time()
            print(f"⬅️ Deleted last letter → Sentence: {sentence}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
