# create_model_files.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

# ---- Step 1. Create a sample dataset ----
data = {
    'Age': np.random.randint(22, 60, 100),
    'Projects Completed': np.random.randint(1, 20, 100),
    'Productivity (%)': np.random.randint(50, 100, 100),
    'Satisfaction Rate (%)': np.random.randint(40, 100, 100),
    'Department': np.random.choice(['HR', 'Finance', 'Engineering', 'Sales'], 100),
    'Position': np.random.choice(['Manager', 'Executive', 'Intern', 'Analyst'], 100),
    'Feedback Score': np.random.uniform(1, 10, 100),
    'Salary': np.random.randint(30000, 120000, 100)
}

df = pd.DataFrame(data)

# ---- Step 2. Encode categorical variables ----
dept_encoder = LabelEncoder()
pos_encoder = LabelEncoder()

df['Department'] = dept_encoder.fit_transform(df['Department'])
df['Position'] = pos_encoder.fit_transform(df['Position'])

# ---- Step 3. Split data ----
X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 4. Train the model ----
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ---- Step 5. Save model and encoders ----
joblib.dump(model, "salary_model.pkl")
joblib.dump(dept_encoder, "Department_encoder.pkl")
joblib.dump(pos_encoder, "Position_encoder.pkl")

print("Files created: salary_model.pkl, Department_encoder.pkl, Position_encoder.pkl")
