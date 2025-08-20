import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import pyttsx3  # Text-to-Speech


class SignLanguageRecognizer:
    def __init__(self):
        # Load model
        model_dict = pickle.load(open('model.p', 'rb'))
        self.model = model_dict['model']

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

        # Webcam
        self.cap = cv2.VideoCapture(0)

        # State
        self.prev_prediction = ""
        self.stable_start_time = time.time()
        self.sentence = ""
        self.prediction_history = deque(maxlen=20)
        self.prev_frame_time = time.time()

    def normalize_landmarks(self, landmarks):
        landmarks = np.array(landmarks).reshape(-1, 3)
        base = landmarks[0]
        landmarks -= base
        max_value = np.max(np.abs(landmarks))
        if max_value != 0:
            landmarks /= max_value
        return landmarks.flatten()

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]), hand_landmarks
        return None, None

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 110)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ TTS Error: {e}")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        frame = cv2.flip(frame, 1)
        landmarks_raw, hand_landmarks = self.extract_landmarks(frame)

        most_common = "?"
        if landmarks_raw is not None:
            normalized = self.normalize_landmarks(landmarks_raw.flatten())
            prediction = self.model.predict([normalized])[0]
            self.prediction_history.append(prediction)
            most_common = Counter(self.prediction_history).most_common(1)[0][0]

            # Stable prediction logic
            if most_common == self.prev_prediction:
                if time.time() - self.stable_start_time >= 4:
                    if most_common.lower() == "space":
                        if self.sentence == "" or self.sentence[-1] != " ":
                            self.sentence += " "
                    else:
                        if self.sentence == "" or self.sentence[-1] != most_common:
                            self.sentence += most_common.upper()
                    self.stable_start_time = time.time()
            else:
                self.prev_prediction = most_common
                self.stable_start_time = time.time()

            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time

        return frame, most_common, self.sentence

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
