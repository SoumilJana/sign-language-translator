import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# ------------------------
# 1. Load combined dataset
# ------------------------
X = np.load('your_landmarks.npy')
y = np.load('your_labels.npy')

# ------------------------
# 2. Stratified train/test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save train/test splits for future experiments
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# ------------------------
# 3. Train model (MLP)
# ------------------------
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
clf.fit(X_train, y_train)

# ------------------------
# 4. Evaluate model
# ------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------
# 5. Save model
# ------------------------
with open('model.p', 'wb') as f:
    pickle.dump({'model': clf}, f)

print("✅ Model saved to model.p")
print("✅ Train/test splits saved as .npy files")
