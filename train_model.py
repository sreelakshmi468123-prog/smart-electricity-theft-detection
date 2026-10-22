import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Load real electricity data
data = pd.read_csv("dataset/G01.csv")

# Select features
X = data[["R", "Y", "B", "AC Status"]].copy()

# Convert to numbers
X = X.apply(pd.to_numeric, errors="coerce")

# Remove missing values
X = X.dropna()

# Train anomaly detection model
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)

# Save model
joblib.dump(model, "model/theft_model.pkl")

print("Model trained successfully!")
print("Number of data points:", len(X))
print("\nExample NORMAL values from real dataset:")
print(X.iloc[0].to_dict())
print("\nMinimum values:")
print(X.min())
print("\nMaximum values:")
print(X.max())
print("\nMean values:")
print(X.mean())