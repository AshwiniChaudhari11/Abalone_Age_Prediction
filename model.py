print("🚀 Starting Abalone Model Training...")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo


# ================= LOAD DATA =================
print("📥 Loading dataset...")
abalone = fetch_ucirepo(id=1)

X = abalone.data.features.copy()
y = abalone.data.targets["Rings"]

# ================= PREPROCESS =================
print("⚙️ Encoding Sex column...")
le = LabelEncoder()
X["Sex"] = le.fit_transform(X["Sex"])

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TRAIN =================
print("🧠 Training model...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ================= PREDICT =================
preds = model.predict(X_test)
errors = y_test - preds

# ================= METRICS =================
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# ================= SAVE FILES =================
joblib.dump(model, "abalone_model.pkl")

joblib.dump({
    "MAE": float(mae),
    "MSE": float(mse),
    "R2": float(r2)
}, "metrics.pkl")

joblib.dump({
    "actual": y_test.tolist(),
    "predicted": preds.tolist(),
    "errors": errors.tolist()
}, "testing_data.pkl")

# ================= VISUALS =================
# ================= VISUALS (USER-FRIENDLY) =================

# ---- 1. Actual vs Predicted Scatter Plot ----
plt.figure(figsize=(7,6))

plt.scatter(y_test, preds, alpha=0.6)

# Ideal line (perfect prediction)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.xlabel("Actual Age (Rings)")
plt.ylabel("Predicted Age (Rings)")
plt.title("Actual vs Predicted Abalone Age")

plt.grid(True)
plt.tight_layout()
plt.savefig("static/actual_vs_predicted.png")


# ---- 2. Error Distribution (Histogram) ----
plt.figure(figsize=(7,6))

plt.hist(errors, bins=30)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Samples")
plt.title("Error Distribution of Predictions")

plt.grid(True)
plt.tight_layout()
plt.savefig("static/error_distribution.png")


# ---- 3. Feature Importance (VERY IMPORTANT FOR UNDERSTANDING) ----
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
plt.barh(features, importance)

plt.xlabel("Importance Score")
plt.title("Feature Importance (Which factors affect age most)")

plt.tight_layout()
plt.savefig("static/feature_importance.png")