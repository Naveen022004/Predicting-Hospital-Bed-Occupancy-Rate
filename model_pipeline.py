import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("Dataset.csv")
df = df.drop(columns=["ICU Bed Source Last Updated"], errors="ignore")
df = df.dropna(subset=["All Bed Occupancy Rate"])

# Columns
numerical_cols = ['Staffed All Beds', 'Staffed ICU Beds', 'Licensed All Beds',
                  'ICU Bed Occupancy Rate', 'Population', 'Population (20+)', 'Population (65+)']
categorical_cols = ['State', 'County Name', 'ICU Bed Source']

# Preprocessing
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Prepare data
X = df[numerical_cols + categorical_cols]
y = df["All Bed Occupancy Rate"]

# Train
model_pipeline.fit(X, y)

# Save model
joblib.dump(model_pipeline, "model.pkl")
print("Model saved as model.pkl")
