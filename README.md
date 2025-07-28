Employee Salary Prediction
A machine learning-powered web application that predicts employee salaries based on factors like age, projects completed, productivity, satisfaction rate, department, position, and feedback score. The system helps HR departments make data-driven and unbiased salary decisions.

Features:

1.Predict employee salaries instantly using an ML model.
2.Streamlit-based interactive web interface with theme customization.
3.Encodes categorical variables (Department, Position) using pre-trained encoders.
4.Trained with an XGBoost Regressor for high accuracy.
5.Saves and loads models using joblib.

Tech Stack:

1.Programming Language: Python 3.x
2.Framework: Streamlit
3.Machine Learning: XGBoost, scikit-learn
4.Data Handling: pandas, numpy
5.Model Persistence: joblib

Requirements:

 Make sure you have the following installed:
1.Python 3.x
2.Streamlit
3.pandas
4.numpy
5.scikit-learn
6.xgboost
7.joblib

How to Run:

Clone this repository:
git clone https://github.com/your-username/employee-salary-prediction.git

Navigate into the project folder:
cd employee-salary-prediction

Run the Streamlit app:
streamlit run app.py

Enter employee details and get instant salary predictions.

Project Structure:

.
├── app.py                   # Streamlit app for predictions
├── create_model_files.py    # Script to train model and save .pkl files
├── salary_model.pkl         # Trained ML model
├── Department_encoder.pkl   # Encoder for Department
├── Position_encoder.pkl     # Encoder for Position
├── scaler.pkl               # (Optional) Scaler for normalization
├── data.csv / emp_data.csv  # Dataset
└── README.md

System Approach:

1.Data Collection and Preprocessing
2.Model Training using XGBoost Regressor
3.Saving model and encoders
4.Building a Streamlit-based web application
5.Predicting salaries in real-time

Deployment:
Can be deployed using Streamlit Cloud, Heroku, or any Python-supported hosting.
