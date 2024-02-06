import cv2
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("face_recognition_model.h5")

# Initialize MTCNN face detector
detector = MTCNN()

# Capture an image from the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect faces using MTCNN
faces = detector.detect_faces(frame)

if faces:
    # Convert frame to RGB (MTCNN expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Loop through each detected face
    for face in faces:
        # Extract the bounding box coordinates
        x, y, w, h = face['box']

        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y+h, x:x+w]

        # Resize the face ROI to match the input size of the model
        face_roi = cv2.resize(face_roi, (150, 150))

        # Preprocess the face image
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension
        face_roi = face_roi / 255.0  # Normalize pixel values

        # Make prediction using the trained model
        predictions = model.predict(face_roi)

        # Print all predictions
        print("Predictions:", predictions)

# Release the camera
cap.release()
