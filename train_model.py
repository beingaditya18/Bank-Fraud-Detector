import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/creditcard.csv')

# Standardize 'Amount' and 'Time'
scaler = StandardScaler()
df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# Train Isolation Forest
X = df.drop('Class', axis=1)
model = IsolationForest(n_estimators=150, contamination=0.0017, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, 'model/isolation_forest.pkl')
print("✅ Model and Scaler saved!")
