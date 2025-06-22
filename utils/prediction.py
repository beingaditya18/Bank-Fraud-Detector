import joblib

def load_model():
    return joblib.load('model/isolation_forest.pkl')

def predict(input_scaled):
    model = load_model()
    result = model.predict(input_scaled)
    return result[0]  # -1 = Fraud, 1 = Not Fraud
