from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ================= LOAD FILES =================
model = joblib.load("abalone_model.pkl")
metrics = joblib.load("metrics.pkl")

HISTORY_FILE = "prediction_history.pkl"

# Load prediction history safely
if os.path.exists(HISTORY_FILE):
    prediction_history = joblib.load(HISTORY_FILE)
else:
    prediction_history = []


# ================= HOME =================
@app.route("/")
def home():
    return render_template(
        "index.html",
        mae=round(metrics["MAE"], 3),
        mse=round(metrics["MSE"], 3),
        r2=round(metrics["R2"], 3),
        prediction_text=None
    )


# ================= ABOUT =================
@app.route("/about")
def about():
    return render_template("about.html")


# ================= PERFORMANCE =================
@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        mae=round(metrics["MAE"], 3),
        mse=round(metrics["MSE"], 3),
        r2=round(metrics["R2"], 3)
    )


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    sex_map = {"M": 0, "F": 1, "I": 2}

    features = np.array([[
        sex_map[request.form["sex"]],
        float(request.form["length"]),
        float(request.form["diameter"]),
        float(request.form["height"]),
        float(request.form["whole_weight"]),
        float(request.form["shucked_weight"]),
        float(request.form["viscera_weight"]),
        float(request.form["shell_weight"])
    ]])

    prediction = model.predict(features)[0]
    age = prediction + 1.5

    # Save prediction history
    prediction_history.append(age)
    joblib.dump(prediction_history, HISTORY_FILE)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Abalone Age: {age:.2f} years",
        mae=round(metrics["MAE"], 3),
        mse=round(metrics["MSE"], 3),
        r2=round(metrics["R2"], 3)
    )


# ================= TESTING DASHBOARD =================
@app.route("/testing")
def testing():

    if not os.path.exists("testing_data.pkl"):
        return "⚠️ Run training script first!"

    data = joblib.load("testing_data.pkl")
    metrics = joblib.load("metrics.pkl")

    return render_template(
        "testing.html",
        mae=float(metrics["MAE"]),
        mse=float(metrics["MSE"]),
        r2=float(metrics["R2"]),
        actual=data["actual"],
        predicted=data["predicted"],
        errors=data["errors"]
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)