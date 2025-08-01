import sys
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import mediapipe as mp
import time
from collections import Counter
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# Load the trained model and class labels
model_path = 'C:/Users/Stephen/Documents/sign_language_proect/Model/sign_model-8-new_data.keras'
model = tf.keras.models.load_model(model_path)  # Replace with your model file path
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Sign Language Interpreter")
        self.image_label = QLabel(self)
        self.prediction_label = QLabel("Prediction: ")
        self.start_button = QPushButton("Start Camera")
        self.end_button = QPushButton("End Sentence")
        self.end_button.setVisible(False)
        self.final_sentence = ""
        self.predicting = False

        self.start_button.clicked.connect(self.start_camera)
        self.end_button.clicked.connect(self.end_sentence)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.end_button)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Buffer and stability trackers
        self.prediction_buffer = []
        self.stable_count = 0
        self.last_prediction = None
        self.last_prediction_time = 0

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.predicting = True

    def end_sentence(self):
        self.predicting = False
        self.prediction_label.setText(f"Final: {self.final_sentence}")
        engine = pyttsx3.init()
        engine.say(self.final_sentence)
        engine.runAndWait()
        self.final_sentence = ""
        self.prediction_buffer.clear()
        self.end_button.setVisible(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Capture User data
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Detect hand landmarks
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Increase bounding box by 40% to account for errors in prediction
            box_width = x_max - x_min
            box_height = y_max - y_min
            expand_x = int(box_width * 0.2)
            expand_y = int(box_height * 0.2)
            x_min = max(x_min - expand_x, 0)
            y_min = max(y_min - expand_y, 0)
            x_max = min(x_max + expand_x, w)
            y_max = min(y_max + expand_y, h)

            roi = frame[y_min:y_max, x_min:x_max]


            if roi.size > 0 and self.predicting:
                #Preprocess image
                orig_h, orig_w = roi.shape[:2]
                desired_size = 224
                scale = min(desired_size / orig_w, desired_size / orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                roi_resized = cv2.resize(roi, (new_w, new_h))
                padded_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
                x_offset = (desired_size - new_w) // 2
                y_offset = (desired_size - new_h) // 2
                padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi_resized

                # Predict on the trained classfier
                roi_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
                img_array = tf.keras.preprocessing.image.img_to_array(roi_rgb)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                prediction = model.predict(img_array)
                confidence = np.max(prediction)
                predicted_class = class_names[np.argmax(prediction)]
                if predicted_class == 'space':
                    predicted_class = ' '
                elif predicted_class == 'del' or predicted_class == 'nothing':
                    predicted_class = ''

                # Stability and debounce mechanism
                if confidence > 0.8:
                    if predicted_class == self.last_prediction:
                        self.stable_count += 1
                    else:
                        self.stable_count = 1
                        self.last_prediction = predicted_class

                    if self.stable_count >= 8 and time.time() - self.last_prediction_time > 1.0:
                        self.final_sentence += predicted_class
                        self.prediction_label.setText(f"Prediction: {self.final_sentence}_")
                        self.prediction_buffer.clear()
                        self.stable_count = 0
                        self.last_prediction_time = time.time()
                        self.end_button.setVisible(True)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
