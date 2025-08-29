import cv2
import mediapipe as mp
import numpy as np
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create data directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Ask user for which letter to record
current_label = input("Enter the label for this gesture (e.g., A, B, space, clear): ").strip().lower()
if not current_label.isalnum():
    print("Invalid label. Please enter an alphanumeric name.")
    exit()

# Set number of samples to collect
samples_per_label = int(input("How many samples do you want to collect? (e.g., 200): "))
collected = 0

# Prepare label directory and starting index
label_dir = os.path.join(DATA_DIR, current_label)
os.makedirs(label_dir, exist_ok=True)
existing_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
start_idx = len(existing_files)

# Start webcam
cap = cv2.VideoCapture(0)
print(f"\n📸 Starting data collection for: {current_label}\nPress 's' to save a sample, 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).flatten()

        cv2.putText(frame, f'{current_label}: {collected}/{samples_per_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No hand detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Collecting Sign Data - Press S to save, Q to quit', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and result.multi_hand_landmarks:
        np.save(os.path.join(label_dir, f'{start_idx + collected}.npy'), landmarks)
        collected += 1
        if collected >= samples_per_label:
            print(f"\n✅ Done collecting {samples_per_label} samples for letter: {current_label}")
            break
    elif key == ord('q'):
        print(f"\n⚠️ Collection manually stopped at {collected} samples for {current_label}")
        break

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Data collection complete! You can now use this data for training your model.")