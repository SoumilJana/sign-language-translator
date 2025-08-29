import cv2
import mediapipe as mp
import numpy as np
import os

# ------------------------
# Setup
# ------------------------
KAGGLE_FOLDER = 'kaggle_asl_dataset'  # folder containing subfolders A-Z and 0-9
OUTPUT_FOLDER = 'kaggle_asl_dataset_npy'  # where .npy files will be saved

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ------------------------
# Function to normalize landmarks
# ------------------------
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist as origin
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value
    return landmarks.flatten()

# ------------------------
# Preprocess images
# ------------------------
for label in os.listdir(KAGGLE_FOLDER):
    label_path = os.path.join(KAGGLE_FOLDER, label)
    if not os.path.isdir(label_path):
        continue

    # Ensure label folder in output
    out_label_dir = os.path.join(OUTPUT_FOLDER, label.lower())  # lowercase for consistency
    if not os.path.exists(out_label_dir):
        os.makedirs(out_label_dir)

    count = 0
    for img_file in os.listdir(label_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks = normalize_landmarks(landmarks)
            np.save(os.path.join(out_label_dir, f"{count}.npy"), landmarks)
            count += 1

    print(f"Processed {count} images for class '{label}'")

print("✅ Kaggle preprocessing complete. All landmarks saved as .npy files.")
