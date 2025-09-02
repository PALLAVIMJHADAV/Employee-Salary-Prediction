# Employee Salary Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-salary-prediction-fpuzgx7ry3e5kravdvpvs3.streamlit.app/)

A machine learning-powered web application that predicts employee salaries based on factors like age, projects completed, productivity, satisfaction rate, department, position, and feedback score. The system helps HR departments make data-driven and unbiased salary decisions.

---

## 🌐 Live Web App

Access the deployed Employee Salary Prediction app here: [Open App](https://employee-salary-prediction-fpuzgx7ry3e5kravdvpvs3.streamlit.app/)

---

## Features

- Predict employee salaries instantly using a pre-trained ML model.
- Streamlit-based interactive web interface with theme customization.
- Encodes categorical variables (Department, Position) using pre-trained encoders.
- Trained with an XGBoost Regressor for high accuracy.
- Saves and loads models using joblib.
- Real-time predictions with minimal latency.

---

## Tech Stack

- **Programming Language:** Python 3.x
- **Framework:** Streamlit
- **Machine Learning:** XGBoost, scikit-learn
- **Data Handling:** pandas, numpy
- **Model Persistence:** joblib

---

## Requirements

Make sure you have the following installed:

- Python 3.x
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

Install missing packages using:

```bash
pip install -r requirements.txt
