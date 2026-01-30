# AIML_app/camera.py
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
import os

class VideoCamera(object):
    def __init__(self):
        # 1. Open Webcam
        self.video = cv2.VideoCapture(0)
        
        # 2. Load Face Detector (Standard OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 3. Load Your Trained Model (.h5 file at root)
        model_path = os.path.join(settings.BASE_DIR, 'emotion_model.h5')
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

        # 4. Emotions List (Must match your Dataset folder names)
        self.emotion_dict = {
            0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
            4: "Neutral", 5: "Sad", 6: "Surprise"
        }

    def __del__(self):
        self.video.release()

    def get_frame_and_label(self):
        success, image = self.video.read()
        if not success:
            return None, None
        
        image = cv2.flip(image, 1)  # Mirror image

        # --- PREPROCESSING ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Low Light Enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # --- DETECTION ---
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        detected_label = "Neutral" # Default

        for (x, y, w, h) in faces:
            # Draw Box
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            
            # Prepare Face for AI (Resize to 48x48)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Predict
            if self.model:
                prediction = self.model.predict(roi_gray, verbose=0)
                max_index = int(np.argmax(prediction))
                detected_label = self.emotion_dict[max_index]
                
                # Draw Text
                cv2.putText(image, detected_label, (x, y-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- ENCODING ---
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), detected_label