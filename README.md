# 📊 AI Customer Churn Prediction Dashboard

An interactive Machine Learning dashboard that predicts whether a customer is likely to churn using multiple Machine Learning algorithms and a Neural Network. The application is built with **Streamlit** and provides an intuitive interface for entering customer information, comparing model predictions, visualizing feature importance, and downloading prediction reports.

---

## 🚀 Live Demo

> https://customer-churn-prediction-model-dashboard.streamlit.app/


---

# 📌 Project Overview

Customer churn prediction is an important business problem where companies identify customers who are likely to discontinue their services. Early prediction allows organizations to take proactive actions to improve customer retention.

This project combines Machine Learning and Streamlit to provide a complete end-to-end customer churn prediction system.

The dashboard allows users to:

- Enter customer information
- Predict churn using multiple Machine Learning models
- Compare predictions across models
- View churn probability
- Analyze customer risk level
- Visualize the most important features affecting prediction
- Download a PDF prediction report

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- ReportLab
- Joblib

---

# 🤖 Machine Learning Models

The dashboard compares predictions from multiple trained models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Neural Network (MLP Classifier)

---

# ✨ Features

✅ Modern Streamlit Dashboard

✅ Interactive Customer Information Form

✅ Real-time Churn Prediction

✅ Multiple Model Comparison

✅ Probability Score

✅ Confidence Score

✅ Overall Customer Risk Analysis

✅ Feature Importance Visualization

✅ PDF Report Generation

✅ Responsive User Interface

---

# 📂 Project Structure

```
AI-Customer-Churn-Prediction-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Models/
│   ├── Logistic Regression.pkl
│   ├── Decision Tree.pkl
│   ├── Random Forest.pkl
│   ├── Support Vector Machine.pkl
│   ├── Neural Network.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── Notebooks
│   └── Customer_Churn_Prediction.ipynb
│
├──Telco-Customer-Churn.csv
│
└── images/
    ├── dashboard.png
    ├── prediction.png

```

---

# 📈 Workflow

```
Customer Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Machine Learning Models
      │
      ▼
Prediction
      │
      ▼
Risk Analysis
      │
      ▼
Feature Importance
      │
      ▼
PDF Report
```

---

# 📊 Input Features

The application uses customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Paperless Billing

---

# 📉 Output

The application predicts:

- Customer Churn (Yes/No)
- Churn Probability
- Confidence Score
- Overall Risk Level
- Feature Importance
- Downloadable Prediction Report

---

# 📷 Screenshots

## Dashboard


![Dashboard](Images/Dashboard.png)

---

## Prediction Results


![Prediction](Images/Prediction.png)



---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Isbah-Ali/Customer-Churn-Prediction-Dashboard.git
```

Move into the project folder

```bash
cd Customer-Churn-Prediction-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📄 Dataset

This project uses the Telco Customer Churn dataset containing customer demographic information, account information, and service details used to predict customer churn.

---

# 📌 Future Improvements

- SHAP Explainability
- XGBoost Model
- LightGBM Integration
- Hyperparameter Optimization
- Database Integration
- User Authentication
- Cloud Deployment
- Real-time Prediction API

---

# 👨‍💻 Author

**Isbah Ali**

GitHub: https://github.com/Isbah-Ali  

LinkedIn: https://linkedin.com/in/isbah-ali-dataanalyst/

---

# ⭐ If you found this project helpful, consider giving it a star!
