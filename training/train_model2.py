import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight

# ------------------------
# Load dataset
# ------------------------
X = np.load('your_landmarks.npy')
y = np.load('your_labels.npy')

print(f"Loaded dataset with {X.shape[0]} samples and {len(np.unique(y))} classes.")

# ------------------------
# Train/test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# ------------------------
# Class weights for imbalance
# ------------------------
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("Class weights:", class_weights)

# ------------------------
# Train MLP
# ------------------------
clf = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
clf.fit(X_train, y_train)

# ------------------------
# Evaluate
# ------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------
# Save model
# ------------------------
with open('model.p', 'wb') as f:
    pickle.dump({'model': clf}, f)

print("\n✅ Model saved to model.p")
