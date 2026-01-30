import cv2
import numpy as np
import tensorflow as tf
import base64
import json
import os
import logging
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# --- LOGGER SETUP ---
logger = logging.getLogger(__name__)

# --- 1. LOAD MODEL GLOBALLY ---
MODEL_PATH = os.path.join(settings.BASE_DIR, 'emotion_model.h5')
mobile_model = None

try:
    # Use "compile=False" to avoid errors if you have custom optimizers in the saved model
    mobile_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Mobile AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Global stats counter (Note: This resets every time the server restarts)
emotion_stats = {'Happy': 0, 'Sad': 0, 'Angry': 0, 'Surprise': 0, 'Neutral': 0, 'Fear': 0}

def index(request):
    return render(request, 'index.html')

# --- PC MODE (Streaming) ---
# We wrap this to prevent Cloud Crashes
def gen(camera):
    while True:
        try:
            frame, label = camera.get_frame_and_label()
            if label:
                # Update global stats safely
                current_count = emotion_stats.get(label, 0)
                emotion_stats[label] = current_count + 1
            
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"Camera Error: {e}")
            break

def video_feed(request):
    try:
        # Import here to avoid "No Camera" crash on startup in Cloud
        from .camera import VideoCamera
        return StreamingHttpResponse(gen(VideoCamera()),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Video Feed Error (Expected on Cloud): {e}")
        return JsonResponse({'error': 'Camera not available on server environment'})

# --- MOBILE MODE (Frame Processing) ---
@csrf_exempt
def process_mobile_frame(request):
    if request.method == 'POST':
        try:
            # 1. Get Image Data
            data = json.loads(request.body)
            image_data = data.get('image')
            
            if not image_data:
                return JsonResponse({'error': 'No image data found'})

            # Remove header if present
            if "base64," in image_data:
                image_data = image_data.split(',')[1]

            # 2. Decode Image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 3. Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            detected_label = "Neutral"

            if len(faces) > 0:
                # LOGIC: Find the largest face (the user close to camera)
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face

                roi_gray = gray[y:y+h, x:x+w]
                
                # Check if model loaded correctly before predicting
                if mobile_model:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray.astype('float32') / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=0)
                    roi_gray = np.expand_dims(roi_gray, axis=-1)

                    prediction = mobile_model.predict(roi_gray, verbose=0)
                    max_index = int(np.argmax(prediction))
                    detected_label = EMOTION_DICT.get(max_index, "Neutral")

            # 4. Update Stats
            if detected_label in emotion_stats:
                emotion_stats[detected_label] += 1
            else:
                 # In case 'Disgust' or other labels appear that aren't in initial dict
                emotion_stats[detected_label] = 1

            return JsonResponse({'emotion': detected_label})
            
        except Exception as e:
            print(f"Mobile Process Error: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid Request'}, status=400)

def get_stats(request):
    return JsonResponse(emotion_stats)