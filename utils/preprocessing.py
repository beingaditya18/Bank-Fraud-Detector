import numpy as np
import joblib

def load_scaler():
    return joblib.load('model/scaler.pkl')

def preprocess_input(time, amount, v_features):
    scaler = load_scaler()
    raw_input = np.array([[time] + v_features + [amount]])
    return scaler.transform(raw_input)
