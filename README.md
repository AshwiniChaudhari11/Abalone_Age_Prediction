# 🐚 Abalone Age Prediction System

An AI-powered web application that predicts the age of abalone using machine learning techniques. The system provides accurate predictions along with a professional dashboard for performance analysis and testing visualization.

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

## 🏗️ Project Structure
abalone-age-prediction/
│
├── app.py # Flask application
├── abalone_model.pkl # Trained ML model
├── metrics.pkl # Model performance metrics
├── testing_data.pkl # Testing data for visualization
│
├── templates/
│ ├── index.html
│ ├── about.html
│ ├── performance.html
│ ├── testing.html
│ └── navbar.html
│
├── static/
│ ├── style.css
│ ├── actual_vs_predicted.png
│ ├── error_distribution.png
│
└── README.md
---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
bash
git clone https://github.com/your-username/abalone-age-prediction.git
cd abalone-age-prediction

2️⃣ Create Virtual Environment (Optional)
conda create -n abalone python=3.10
conda activate abalone

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Training Script
python train.py

5️⃣ Run Flask App
python app.py

6️⃣ Open in Browser
http://127.0.0.1:5000/

📊 Testing & Visualization
The system includes an advanced testing dashboard with:
📌 Scatter Plot (Actual vs Predicted)
📌 Error Distribution Graph
📌 Accuracy Gauge (R² Score)

These visualizations help in understanding model performance clearly.

📈 Model Evaluation Metrics
MAE (Mean Absolute Error) → Measures average prediction error
MSE (Mean Squared Error) → Penalizes large errors
R² Score → Indicates model accuracy

💡 How It Works
User enters abalone measurements
Data is processed and passed to trained ML model
Model predicts number of rings

Age is calculated using formula:
Age = Rings + 1.5
Results are displayed with performance insights

🛠️ Technologies Used
Python
Flask
Scikit-learn
Pandas & NumPy
Matplotlib
Chart.js (for frontend graphs)

🔮 Future Improvements
📱 Mobile responsive UI
☁️ Cloud deployment (AWS/Render)
🤖 Advanced ML models
📊 Real-time prediction analytics