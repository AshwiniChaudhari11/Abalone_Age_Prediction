
# 🐚 Abalone Age Prediction System

An AI-powered web application that predicts the age of abalone using machine learning and provides visual performance analysis.

---

## 🚀 Features

- 🔍 Predict abalone age based on physical measurements  
- 📊 Model performance dashboard (MAE, MSE, R² Score)  
- 📈 Interactive testing dashboard with visual graphs  
- 🎯 Accuracy gauge visualization  
- 📉 Error distribution analysis  
- 🌐 User-friendly web interface using Flask  

---

## 🧠 Machine Learning Model

- Algorithm: **Random Forest Regressor**
- Dataset: **UCI Abalone Dataset**
- Features Used:
  - Sex
  - Length
  - Diameter
  - Height
  - Whole Weight
  - Shucked Weight
  - Viscera Weight
  - Shell Weight

---

## 📁 Project Structure

```

abalone-age-prediction/
│
├── app.py                  # Flask application
├── abalone_model.pkl       # Trained ML model
├── metrics.pkl             # Model performance metrics
├── testing_data.pkl        # Testing data
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── performance.html
│   ├── testing.html
│   └── navbar.html
│
├── static/
│   ├── style.css
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│
└── README.md

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/abalone-age-prediction.git
cd abalone-age-prediction
````

### 2️⃣ Create Virtual Environment (Optional)

```bash
conda create -n abalone python=3.10
conda activate abalone
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Training Script

```bash
python train.py
```

### 5️⃣ Run Flask App

```bash
python app.py
```

### 6️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 Testing & Visualization

The system includes an advanced testing dashboard with:

* 📌 Scatter Plot (Actual vs Predicted)
* 📌 Error Distribution Graph
* 📌 Accuracy Gauge (R² Score)

---

## 🧠 Model Details

* Algorithm: Random Forest Regressor
* Dataset: UCI Abalone Dataset

---

## 📈 Evaluation Metrics

* MAE – Mean Absolute Error
* MSE – Mean Squared Error
* R² Score – Model Accuracy

---

## 💡 How It Works

1. User enters input values
2. Data is processed
3. Model predicts rings
4. Age is calculated
5. Results are displayed with graphs

---


````
