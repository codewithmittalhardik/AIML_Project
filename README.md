# 🧠 NeuroSense AI

**NeuroSense AI** is a real-time emotion detection web application built with **Django** and **TensorFlow**. It uses a Convolutional Neural Network (CNN) to analyze facial expressions via a webcam or mobile phone camera and classifies them into emotions (Happy, Sad, Angry, Surprise, Neutral, Fear) instantly.

## 🚀 Features

- **Real-Time Detection:** Live video processing using OpenCV and TensorFlow.
- **Dual Camera Support:** Use your laptop webcam OR connect your mobile phone camera remotely.
- **Live Analytics:** Dynamic Pie Chart (Chart.js) showing emotion distribution during the session.
- **Event Logging:** Real-time log of dominant emotions with timestamps.
- **Responsive UI:** Modern glassmorphism design built with Tailwind CSS.

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Django 5.0
- **AI/ML:** TensorFlow (Keras), OpenCV, NumPy
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Chart.js)
- **Deployment:** Render / Gunicorn

## 📦 Installation (Local)

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/neurosense-ai.git](https://github.com/YOUR_USERNAME/neurosense-ai.git)
cd neurosense-ai
```
# 🧠 NeuroSense AI

**NeuroSense AI** is a real-time emotion detection web application built with **Django** and **TensorFlow**. It uses a Convolutional Neural Network (CNN) to analyze facial expressions via a webcam or mobile phone camera and classifies them into emotions (Happy, Sad, Angry, Surprise, Neutral, Fear) instantly.

![Project Banner](https://via.placeholder.com/1000x300?text=NeuroSense+AI+Dashboard)

## 🚀 Features

- **Real-Time Detection:** Live video processing using OpenCV and TensorFlow.
- **Dual Camera Support:** Use your laptop webcam OR connect your mobile phone camera remotely.
- **Live Analytics:** Dynamic Pie Chart (Chart.js) showing emotion distribution during the session.
- **Event Logging:** Real-time log of dominant emotions with timestamps.
- **Responsive UI:** Modern glassmorphism design built with Tailwind CSS.

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Django 5.0
- **AI/ML:** TensorFlow (Keras), OpenCV, NumPy
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Chart.js)
- **Deployment:** Render / Gunicorn

## 📦 Installation (Local)

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/neurosense-ai.git](https://github.com/YOUR_USERNAME/neurosense-ai.git)
cd neurosense-ai
```
**Create a Virtual Environment**
## Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
## Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
**Install Dependencies**
```text
pip install -r requirements.txt
```
**Start the Server**
```text
python manage.py runserver
```
**Project Structure**
```text
neurosense-ai/
├── AIML_PROJECT/        # Main Django Configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── emotion_app/         # App Logic (Views & Models)
│   ├── templates/       # HTML Files
│   ├── views.py         # Detection Logic
│   └── model.h5         # Trained Emotion Detection Model
├── static/              # CSS, JS, Images
├── manage.py
├── requirements.txt
└── README.md
```
