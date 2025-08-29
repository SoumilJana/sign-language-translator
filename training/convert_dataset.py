import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # use wrist as origin
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value  # scale to range [-1, 1]
    return landmarks.flatten()

DATA_DIR = 'data'  # Folder for your webcam-collected data
KAGGLE_DIR = 'kaggle_data'  # Folder containing preprocessed Kaggle landmarks (.npy)

landmark_data = []
labels = []

# ------------------------
# 1. Load webcam data
# ------------------------
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    print(f"Loading {len(files)} samples for class '{label}' from webcam data.")
    for file in files:
        data = np.load(os.path.join(label_dir, file))
        if np.isnan(data).any() or np.isinf(data).any():
            continue
        if data.shape != (21*3,):  # expected shape
            continue
        normalized = normalize_landmarks(data)
        landmark_data.append(normalized)
        labels.append(label)

# ------------------------
# 2. Load Kaggle data if exists
# ------------------------
if os.path.exists(KAGGLE_DIR):
    for file in os.listdir(KAGGLE_DIR):
        if file.endswith('.npy'):
            # Expect file names like 'A_001.npy', 'B_015.npy' or separate folder per label
            data = np.load(os.path.join(KAGGLE_DIR, file))
            if np.isnan(data).any() or np.isinf(data).any():
                continue
            if data.shape != (21*3,):
                continue
            normalized = normalize_landmarks(data)
            # Extract label from file name (assuming first character is label)
            label = os.path.basename(file).split('_')[0].lower()
            landmark_data.append(normalized)
            labels.append(label)
    print(f"Added Kaggle data. Total samples now: {len(landmark_data)}")

# Convert to numpy arrays
landmark_data = np.array(landmark_data)
labels = np.array(labels)

# ------------------------
# 3. Label encoding
# ------------------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
np.save('label_classes.npy', le.classes_)

# ------------------------
# 4. Shuffle dataset
# ------------------------
X, y = shuffle(landmark_data, labels_encoded, random_state=42)

# ------------------------
# 5. Save final arrays
# ------------------------
np.save('your_landmarks.npy', X)
np.save('your_labels.npy', y)

# Print summary
unique_classes = len(set(labels))
print(f"✅ Saved {len(X)} samples for {unique_classes} classes (after normalization, encoding, shuffling).")
