import pickle
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from spellchecker import SpellChecker  

from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout

from kivymd.uix.dialog import MDDialog  
from kivymd.uix.button import MDFlatButton, MDRaisedButton  

# Load the UI design from the .kv file
Builder.load_file("ui.kv")

class CameraScreen(BoxLayout):
    """
    Main widget that handles camera feed, hand detection,
    sign prediction, and sentence construction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # --- Bind Keyboard Listener ---
        Window.bind(on_key_down=self._on_key_down)

        # --- Autocorrect Setup ---
        try:
            self.spell = SpellChecker()
            self.use_autocorrect = True
            print("✅ Spellchecker loaded.")
        except Exception as e:
            print(f"⚠️ Could not load spellchecker: {e}")
            self.spell = None
            self.use_autocorrect = False

        # --- State variables for sign detection logic ---
        self.sentence = ""
        self.prediction_history = deque(maxlen=15)
        self.stable_letter = None
        self.hold_start_time = None
        self.is_camera_active = False

        # --- NEW: Threading variable ---
        self.latest_frame = None
        
        # --- NEW: TTS Lock to prevent crashes ---
        self.tts_lock = threading.Lock()
        
        # --- NEW: Logic Lock to prevent race conditions ---
        self.logic_lock = threading.RLock()

        # --- Constants ---
        # How long to hold a sign before it's added
        self.HOLD_TIME_TO_ADD = 1.2 

        # --- Load ML Model and Labels ---
        try:
            model_data = pickle.load(open("app/assets/model.p", "rb"))
            self.model = model_data["model"]
            self.label_classes = np.load("app/assets/label_classes.npy", allow_pickle=True)
            print("✅ Model and labels loaded successfully.")
        except FileNotFoundError:
            print("❌ Error: 'model.p' or 'label_classes.npy' not found.")
            print("Ensure the path matches your folder structure.")
            self.model = None

        # --- Initialize MediaPipe Hands ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        print("✅ MediaPipe Hands initialized.")

        # --- Text-to-Speech Engine ---
        print("✅ Text-to-Speech engine ready.")

        # --- Initialize OpenCV Video Capture ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam.")
            Clock.schedule_once(lambda dt: self._set_camera_status_text("Error: Webcam not found.", True))
        else:
            print("✅ Webcam opened successfully.")
            self.is_camera_active = True
            
            Clock.schedule_once(lambda dt: self._set_camera_status_text("Loading Model...", True))

            threading.Thread(target=self.cv_processing_loop, daemon=True).start()
            Clock.schedule_interval(self.update_ui, 1.0 / 30.0)

    def _on_key_down(self, instance, keyboard, keycode, text, modifiers):
        """
        Handles keyboard shortcuts for app controls.
        """
        if keycode == 40 or keycode == 271:
            self.trigger_speak()
        elif keycode == 42:
            self.delete_last_letter()
        return True

    # --- MODIFIED: Autocorrect Function (forces lowercase for better matching) ---
    def _autocorrect_text(self, text: str) -> str:
        """
        Corrects the spelling of a given string.
        """
        if not self.use_autocorrect or not self.spell:
            return text
        
        # Force lowercase for better spell.correction() matching
        words = text.lower().split() 
        corrected = []
        for w in words:
            corrected_word = self.spell.correction(w)
            corrected.append(corrected_word or w)
        # Returns a lowercase string
        return " ".join(corrected)

    def update_ui(self, dt):
        """
        This method is called by the Kivy Clock 30x/sec.
        Its ONLY job is to draw the latest available frame to the screen.
        """
        if self.latest_frame is not None:
            self._display_frame(self.latest_frame)

    def cv_processing_loop(self):
        """
        This method runs in a separate thread.
        It handles all the heavy webcam reading, landmark detection,
        and ML prediction logic.
        """
        is_first_frame = True
        
        while self.is_camera_active:
            if not self.model:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._set_camera_status_text("Webcam disconnected.", True)
                time.sleep(0.5)
                continue

            if is_first_frame:
                self._set_camera_status_text("", False)
                is_first_frame = False

            frame = cv2.flip(frame, 1)
            
            landmarks, hand_landmarks_for_drawing = self._extract_landmarks(frame)

            if landmarks is not None:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks_for_drawing, self.mp_hands.HAND_CONNECTIONS)
                normalized_landmarks = self._normalize_landmarks(landmarks)

                confidence = 100.0
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba([normalized_landmarks])[0]
                    prediction_int = np.argmax(probabilities)
                    confidence = probabilities[prediction_int] * 100.0
                else:
                    prediction_int = self.model.predict([normalized_landmarks])[0]
                
                predicted_letter = self.label_classes[prediction_int]

                self._update_sentence(predicted_letter)
                self._draw_prediction_on_frame(frame, predicted_letter, confidence, hand_landmarks_for_drawing)
            else:
                with self.logic_lock:
                    if self.stable_letter is not None: 
                        self.stable_letter = None
                        self.hold_start_time = None
                        self.prediction_history.clear()
                        self._set_progress_bar_value(0) 

            self._draw_sentence_on_frame(frame)
            
            self.latest_frame = frame.copy()
            
            time.sleep(0.01)

    def _extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            return landmarks, hand_landmarks
        return None, None
        
    def _normalize_landmarks(self, landmarks):
        landmarks = landmarks.copy()
        base = landmarks[0]
        landmarks -= base
        max_value = np.max(np.abs(landmarks))
        if max_value > 1e-6:
            landmarks /= max_value
        return landmarks.flatten()

    @mainthread
    def _set_camera_status_text(self, text, is_visible):
        """
        Safely updates the camera status label from any thread.
        Hides/shows the camera view accordingly.
        """
        if self.ids.cam_status_label:
            self.ids.cam_status_label.text = text
            self.ids.cam_status_label.opacity = 1.0 if is_visible else 0.0
            self.ids.cam_view.opacity = 0.0 if is_visible else 1.0

    @mainthread
    def _set_progress_bar_value(self, value):
        """
        Safely updates the hold progress bar from any thread.
        """
        if self.ids.hold_progress:
            self.ids.hold_progress.value = value

    @mainthread
    def _set_sentence_label(self, text):
        """
        Safely updates the Kivy label from any thread.
        """
        if self.ids.sentence_label:
            self.ids.sentence_label.text = text

    def _update_sentence(self, predicted_letter):
        """
        This logic handles the hold-to-add-letter feature
        and now also updates the progress bar.
        """
        with self.logic_lock:
            self.prediction_history.append(predicted_letter)
            try:
                most_common_prediction = Counter(self.prediction_history).most_common(1)[0][0]
            except IndexError:
                self._set_progress_bar_value(0) 
                return 

            if most_common_prediction != self.stable_letter:
                self.stable_letter = most_common_prediction
                self.hold_start_time = time.time()  
                self._set_progress_bar_value(0) 
            
            elif self.hold_start_time: 
                elapsed = time.time() - self.hold_start_time
                progress_percent = (elapsed / self.HOLD_TIME_TO_ADD) * 100
                
                self._set_progress_bar_value(min(progress_percent, 100))

                if elapsed >= self.HOLD_TIME_TO_ADD:
                    
                    if self.stable_letter == 'space':
                        words = self.sentence.strip().split()
                        if words:
                            last_word = words[-1]
                            corrected_word = self._autocorrect_text(last_word)
                            # --- MODIFIED: Convert back to UPPERCASE for display ---
                            words[-1] = corrected_word.upper() 
                            self.sentence = " ".join(words)
                        
                        self.sentence += ' ' 
                    
                    elif self.stable_letter: 
                        self.sentence += self.stable_letter.upper()
                    
                    self._set_sentence_label(self.sentence) 
                    
                    self.hold_start_time = None
                    self.prediction_history.clear()
                    self._set_progress_bar_value(0) 


    def _draw_prediction_on_frame(self, frame, letter, confidence, hand_landmarks):
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
        text = f"{letter.upper()} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x_min, max(30, y_min - 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_sentence_on_frame(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        with self.logic_lock:
            text_to_draw = self.sentence
        
        cv2.putText(frame, f'Live: {text_to_draw}', (20, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    def _display_frame(self, frame):
        """
        This is now called by update_ui (main thread), so it's safe.
        """
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_view.texture = texture

    def trigger_speak(self):
        with self.logic_lock:
            if not self.sentence.strip():
                return
            
            sentence_to_speak = self.sentence
            # Autocorrect the *final* sentence before speaking
            sentence_to_speak = self._autocorrect_text(sentence_to_speak)
        
        if not self.tts_lock.acquire(blocking=False):
            print("🟡 TTS already in progress. Ignoring new request.")
            return
        
        print(f"🟢 Speaking: '{sentence_to_speak}'")
        threading.Thread(target=self._speak_thread_safe, args=(sentence_to_speak,), daemon=True).start()


    def _speak_thread_safe(self, sentence_to_speak):
        """
        Initializes the TTS engine *inside* this thread
        and clears the sentence after speaking.
        """
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 125)
            tts_engine.say(sentence_to_speak)
            tts_engine.runAndWait()
            tts_engine.stop()
            
            self.clear_sentence()
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            
        finally:
            self.tts_lock.release()

    def delete_last_letter(self):
        """
        Deletes the last character from the sentence.
        """
        with self.logic_lock:
            if self.sentence:
                self.sentence = self.sentence[:-1]
                
                if not self.sentence:
                    self._set_sentence_label("Hold a sign to begin")
                else:
                    self._set_sentence_label(self.sentence)
                print("Deleted last letter.")

    def clear_sentence(self):
        """
        Clears the sentence and resets the hold logic.
        """
        with self.logic_lock:
            self.sentence = ""
            self.prediction_history.clear()
            self.stable_letter = None
            self.hold_start_time = None
            
            self._set_sentence_label("Hold a sign to begin")
            self._set_progress_bar_value(0)
            print("Cleared sentence.")
        
    def stop_camera(self):
        self.is_camera_active = False
        time.sleep(0.1) 
        
        if self.cap:
            self.cap.release()
            print("Webcam released.")
        if self.hands:
            self.hands.close()
            print("MediaPipe Hands closed.")


class SignTranslatorApp(MDApp):
    
    # --- DELETED: dialog = None ---

    def build(self):
        Window.icon = "app/assets/icon.ico"
        self.title = "SignVision"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Cyan"
        self.theme_cls.accent_palette = "Amber"

        # --- DELETED: Window.bind(on_request_close = self.confirm_exit_dialog) ---

        return CameraScreen()
    
    # --- DELETED: confirm_exit_dialog method ---

    # --- DELETED: perform_exit method ---

    def on_stop(self):
        print("🛑 App is stopping.")
        if self.root: # Ensure root exists
            self.root.stop_camera()
            # --- Unbind Keyboard Listener ---
            Window.unbind(on_key_down = self.root._on_key_down)
        
        # --- DELETED: Window.unbind(on_request_close = self.confirm_exit_dialog) ---


if __name__ == "__main__":
    SignTranslatorApp().run()