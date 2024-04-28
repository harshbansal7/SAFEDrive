import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load the Keras model
model = load_model('eye_detector.h5')

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the face image
def preprocess_face(face):
    # Resize face to 100x100
    face = cv2.resize(face, (100, 100))
    # Convert face to grayscale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values to be between 0 and 1
    face = face / 255.0
    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    # Add channel dimension (Keras expects inputs with shape (batch_size, height, width, channels))
    face = np.expand_dims(face, axis=3)
    return face

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face image
        preprocessed_face = preprocess_face(face)

        # Make prediction using the model
        output = model.predict(preprocessed_face)

        # Print the output
        print("Model Output:", output)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
