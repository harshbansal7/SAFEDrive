import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import math
import time

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)

total = 0

# pre_calculations
width = 320
height = 240
interpolation_technique = cv2.INTER_AREA

config = {
    'SCREEN_PRINTS' : True,
    'FPS' : False,
    'LANDMARKS' : True,
    'LINE' : True,
}

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)  # ear = eye

    # return the eye aspect ratio
    return ear

def calculate_roll_index(left_eye_outer, right_eye_outer):
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    angle = math.atan2(dy, dx)

    # Normalize the angle to the range [-pi/2, pi/2]
    angle_normalized = angle / (math.pi / 2)

    return angle_normalized

camera = cv2.VideoCapture(0)

FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_PREDICTOR = dlib.shape_predictor("model_full_mouth_jmd-2.dat")

while True:
    ret, frame = camera.read()

    if ret == False:
        print(
            "Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n"
        )
        break

    frame= cv2.resize(frame, (width, height), interpolation_technique)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_DETECTOR(frame_gray, 0)

    if len(faces) > 0:
        for k, face in enumerate(faces):
            shape = LANDMARK_PREDICTOR(frame_gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[0:6]
            rightEye = shape[6:12]
            mouth = shape[12:]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            ear = (leftEAR + rightEAR) / 2.0
            
            if ear > 0.28:
                total = 0
                GPIO.output(4, GPIO.LOW)
                if config['SCREEN_PRINTS']:
                    cv2.putText(frame, f"Eyes Open | EAR = {round(ear, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                total += 1
                if total > 15:
                    GPIO.output(4, GPIO.HIGH)
                    if config['SCREEN_PRINTS']:
                        cv2.putText(frame, f"Drowsiness Detected", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if config['SCREEN_PRINTS']:
                    cv2.putText(frame, f"Eyes Closed | EAR = {round(ear, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


            if config['LINE']:
                cv2.line(frame, shape[0], shape[9], (0, 255, 0), 2, 8)
                cv2.putText(frame, f"Roll Index : {round(calculate_roll_index(shape[0], shape[9]), 3)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if config['LANDMARKS']:
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
    else:
        GPIO.output(4, GPIO.LOW)
    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        camera.release()
        break
