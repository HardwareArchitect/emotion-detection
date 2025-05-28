import cv2
from keras.models import model_from_json
import numpy as np
import pyttsx3
import time

# Load model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())

model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Initialize webcam
webcam = cv2.VideoCapture(0)

last_spoken = ""
last_time = time.time()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = extract_features(roi_gray)
        prediction = model.predict(img)
        label = labels[int(np.argmax(prediction))]

        # Speak if label changed or 5+ seconds passed
        current_time = time.time()
        if label != last_spoken or current_time - last_time > 5:
            speak(f"You seem {label}")
            last_spoken = label
            last_time = current_time

        # Draw rectangle and custom label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        display_text = f"You seem {label} 😊"
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
