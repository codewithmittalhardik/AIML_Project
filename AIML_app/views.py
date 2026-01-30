import cv2
import numpy as np
import tensorflow as tf
import base64
import json
import os
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .camera import VideoCamera

# --- 1. LOAD MODEL GLOBALLY (For Mobile Speed) ---
# We load it here once so we don't reload it for every single frame sent from the phone.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'emotion_model.h5')
try:
    mobile_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Mobile AI Model Loaded!")
except:
    print("❌ Could not load model for mobile view.")
    mobile_model = None

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Global stats counter
emotion_stats = {}

def index(request):
    return render(request, 'index.html')

# --- PC MODE (Streaming) ---
def gen(camera):
    while True:
        frame, label = camera.get_frame_and_label()
        if label:
            emotion_stats[label] = emotion_stats.get(label, 0) + 1
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# --- MOBILE MODE (Frame Processing) ---
@csrf_exempt
def process_mobile_frame(request):
    if request.method == 'POST':
        try:
            # 1. Get Image Data (Base64)
            data = json.loads(request.body)
            image_data = data['image'].split(',')[1] # Remove the "data:image/jpeg..." header
            
            # 2. Decode to Image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 3. Preprocessing (Same as Camera.py)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            detected_label = "Neutral"
            
            # 4. Predict
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                if mobile_model:
                    prediction = mobile_model.predict(roi_gray, verbose=0)
                    max_index = int(np.argmax(prediction))
                    detected_label = EMOTION_DICT[max_index]
            
            # 5. Update Stats
            if detected_label != "Neutral":
                emotion_stats[detected_label] = emotion_stats.get(detected_label, 0) + 1

            return JsonResponse({'emotion': detected_label})
            
        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid Request'})

def get_stats(request):
    return JsonResponse(emotion_stats)