import pandas as pd
import joblib as jb
import heartpy as hp
import RPi.GPIO as GPIO
import time
import math
import numpy as np 

from sklearn.preprocessing import MinMaxScaler

import max30102

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT) #11 pin


# Load the trained model
MODEL = jb.load('heart_rate_latest.joblib')
SCALER = jb.load('scaler_feature1.joblib')

# Selected features for prediction
features = ['bpm', 'ibi', 'breathingrate']

# Function to remove NaN values
def remove_nan(val):
    if math.isnan(val):
        return 0
    return val

# Function to process PPG data
def process_ppg(data):
    sr = 100 # Sampling rate of 50-100 sample per second
    
    data = np.array(data).reshape(-1, 1)
    
    scaled_data = SCALER.transform(data)
    
    print("SCALED", len(data), data.shape, data)
    
    filtered = hp.filter_signal(scaled_data.reshape(1, -1), [0.5, 15], sample_rate=sr, order=3, filtertype='bandpass')
    
    print("FILTERED", filtered, filtered.shape)
    
    working_data, measures = hp.process(filtered.flatten(), sample_rate=sr)
    print("BPM PRINT", measures['bpm'])
    
    return working_data, measures
    

# Function to select features for prediction
def pick_features(all_measures, selected):
    print(all_measures['bpm'], selected)
    x_train = []
    #for i in range(len(all_measures['bpm'])):
    row = []
    for cat in selected:              
        value = all_measures[cat]
        row.append(remove_nan(value))
        x_train.append(row)
        
    return x_train

# Function to predict heart rate
def predict(inputArr):
    w, mp = process_ppg(inputArr)
    x = pick_features(mp, features)
    y = MODEL.predict(x)
    return y[0]

# Function to read PPG data from the sensor connected to GPIO
def read_sensor_data():
    data = []
    duration = 8

    m = max30102.MAX30102()

    while True:
        # Read analog data from sensor
        red, ir = m.read_sequential()
        data += red
        
        # Keep the data within the window size (800 entries)
        if len(data) > 800:
            data = data[-800:]  # Keep the last 800 entries
        
        # If the data reaches the desired duration, break the loop
        if len(data) >= duration * 100:
            break
            
        # Wait for 1 second before reading more data
        time.sleep(1)

    return data

# Main function to record data from the sensor and make predictions
def main():
    recording_duration = 8  # Duration of recording in seconds
    
    while True:
        sensor_data = read_sensor_data()
        if sensor_data:
            heart_rate_prediction = predict(sensor_data)
            print("Predicted Status:", (heart_rate_prediction > 3.5))

