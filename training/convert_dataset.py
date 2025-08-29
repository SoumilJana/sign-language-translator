import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# ------------------------
# Settings
# ------------------------
DATA_DIR = 'data'  # Webcam-collected data
KAGGLE_DIR = 'kaggle_asl_dataset_npy'  # Preprocessed Kaggle .npy folder
NORMALIZE_KAGGLE = False  # Set False if Kaggle .npy are already normalized

# ------------------------
# Normalize landmarks function
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
# Load dataset
# ------------------------
landmark_data = []
labels = []

def load_npy_folder(folder_path, normalize=True):
    total_count = 0
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        print(f"Loading {len(files)} samples for class '{label}' from '{folder_path}'")
        count = 0
        for file in files:
            data = np.load(os.path.join(label_path, file))
            if np.isnan(data).any() or np.isinf(data).any():
                continue
            if data.shape != (21*3,):
                continue
            if normalize:
                data = normalize_landmarks(data)
            landmark_data.append(data)
            labels.append(label.lower())
            count += 1
        total_count += count
    return total_count

# Webcam data
webcam_count = load_npy_folder(DATA_DIR, normalize=True)

# Kaggle data
kaggle_count = 0
if os.path.exists(KAGGLE_DIR):
    kaggle_count = load_npy_folder(KAGGLE_DIR, normalize=NORMALIZE_KAGGLE)

print(f"\n✅ Total samples loaded: Webcam={webcam_count}, Kaggle={kaggle_count}, Total={webcam_count + kaggle_count}")

# ------------------------
# Convert to numpy arrays
# ------------------------
landmark_data = np.array(landmark_data)
labels = np.array(labels)

# ------------------------
# Label encoding
# ------------------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
np.save('label_classes.npy', le.classes_)

# ------------------------
# Shuffle dataset
# ------------------------
X, y = shuffle(landmark_data, labels_encoded, random_state=42)

# ------------------------
# Save final arrays
# ------------------------
np.save('your_landmarks.npy', X)
np.save('your_labels.npy', y)

# ------------------------
# Print summary
# ------------------------
unique_classes = len(set(labels))
print(f"✅ Final dataset saved with {len(X)} samples for {unique_classes} classes (letters + numbers)")
for cls in le.classes_:
    count = np.sum(labels == cls)
    print(f"Class '{cls}': {count} samples")
