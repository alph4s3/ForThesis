# 🌡️ ImpactGuard: Extreme Heat Early Warning System

**ImpactGuard** is a Streamlit-based decision-support system designed to enhance extreme heat prediction using **impact-centric variables** such as Wet-bulb Temperature, Urban Heat Island (UHI) Intensity, and PM2.5 levels.

Developed as an undergraduate thesis project for **BS Computer Science**, this system demonstrates how integrating environmental and physiological factors can improve heat risk forecasting beyond traditional weather-based models.

---

## 📌 Overview

Extreme heat events pose increasing risks to public health, especially in urban environments. Traditional forecasting methods rely primarily on temperature and humidity, which may not fully capture human heat stress.

**ImpactGuard addresses this gap by:**
- Incorporating **impact-centric variables**
- Providing **interpretable predictions** using explainable AI
- Delivering **actionable alerts** for local decision-makers

---

## 🚀 Key Features

### 🔮 Heat Risk Prediction
- Forecasts extreme heat risk for up to **7 days**
- Uses a simulated **LSTM-based model pipeline**
- Outputs probability-based risk levels:
  - 🟢 Low
  - 🟡 Moderate
  - 🟠 High
  - 🔴 Extreme

---

### ⚖️ Model Comparison
- Compares:
  - **Baseline Model** (Temperature, Humidity, Wind Speed)
  - **Impact-Centric Model** (+ UHI, Wet-bulb, PM2.5)
- Displays performance metrics:
  - RMSE, MAE, Accuracy, F1 Score, AUC-ROC

---

### 🔍 Explainable AI (XAI)
- Uses **SHAP (SHapley Additive Explanations)**
- Identifies top contributing features
- Provides:
  - Feature importance visualization
  - Natural language explanations

---

### 🚨 Alert System
- Generates risk-based alerts with:
  - Severity classification
  - Recommended actions
- Supports public health decision-making

---

### 📊 Interactive Dashboard
- Built using **Streamlit + Plotly**
- Includes:
  - Forecast tables
  - Trend graphs
  - Risk probability charts
  - Location mapping

---

## 🧠 System Architecture

User Input (Location, Days, Model)  
        ↓  
Streamlit Frontend (UI)  
        ↓  
Data Fetch Module (Weather + Impact Variables)  
        ↓  
Prediction Module (LSTM Model)  
        ↓  
Explainability Module (SHAP)  
        ↓  
Visualization & Alerts  

---

## ⚙️ Technologies Used

- Python 3.11
- Streamlit
- Pandas / NumPy
- Plotly
- SHAP

---

## 📂 Project Structure

ForThesis/  
│  
├── app.py                # Main Streamlit application  
├── requirements.txt     # Dependencies  
└── README.md            # Project documentation  

> Note: Backend modules (DataPipeline, LSTMModel, etc.) are structured as placeholders for integration with a full machine learning pipeline.

---

## ▶️ How to Run

1. Clone the repository  
   git clone https://github.com/alph4s3/ForThesis.git  
   cd ForThesis  

2. Create a virtual environment  
   python -m venv venv  
   source venv/bin/activate   (Mac/Linux)  
   venv\Scripts\activate      (Windows)  

3. Install dependencies  
   pip install -r requirements.txt  

4. Run the app  
   streamlit run app.py  

---

## ⚠️ Limitations

- Current implementation uses **simulated data and model outputs** for demonstration purposes.
- The system is designed with a **modular backend architecture**, allowing integration of real datasets and trained LSTM models in future work.
- External data sources such as ERA5 and OpenAQ are planned for full deployment.

---

## 🔮 Future Improvements

- Integration of real-time APIs (ERA5, OpenAQ)
- Deployment of trained LSTM models
- Expansion to more cities and regions
- Mobile-responsive version
- Automated alert notifications (SMS / Email)

---

## 👨‍💻 Author

John Wallace Aceres  
BS Computer Science  
Ateneo de Zamboanga University  

---

## 📜 Thesis Title

"Enhancing Extreme Heat Prediction Using Impact-Centric Variables in Machine Learning Models"

---

## 💡 Final Note

ImpactGuard is not just a predictive tool — it is a **decision-support system** designed to bridge the gap between data science and public health action in the face of climate change.
