import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. CLEAN DATA
# ==========================================

df = df.drop(
    columns=["ICU Bed Source Last Updated"],
    errors="ignore"
)

df = df.dropna(
    subset=["All Bed Occupancy Rate"]
)


# ==========================================
# 3. DEFINE FEATURES
# ==========================================

numerical_cols = [
    "Staffed All Beds",
    "Staffed ICU Beds",
    "Licensed All Beds",
    "ICU Bed Occupancy Rate",
    "Population",
    "Population (20+)",
    "Population (65+)"
]

categorical_cols = [
    "State",
    "County Name",
    "ICU Bed Source"
]


# ==========================================
# 4. SELECT X AND Y
# ==========================================

X = df[numerical_cols + categorical_cols]

y = df["All Bed Occupancy Rate"]


# ==========================================
# 5. NUMERICAL PIPELINE
# ==========================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ==========================================
# 6. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# ==========================================
# 7. PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        numerical_cols
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_cols
    )
])


# ==========================================
# 8. MODEL
# ==========================================

model_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("Training model...")

model_pipeline.fit(X, y)


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model_pipeline,
    "model.pkl"
)

print("====================================")
print("✅ Model trained successfully!")
print("✅ model.pkl created successfully!")
print("====================================")
