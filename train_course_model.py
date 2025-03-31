import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# ✅ Load dataset
file_path = r"D:\C_F_O.model\course_data_fixed.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")

df = pd.read_csv(file_path)

# ✅ Define feature categories
categorical_features = ["Difficulty_Level"]
numerical_features = [
    "Course_Duration", "Num_Modules", "Certification", "Competitor_Price",
    "Student_Demand", "Dropout_Rate", "Feedback_Score", "Marketing_Spend", "Discount_Offered"
]

# ✅ One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

# ✅ Define X and y
X = df.drop(columns=["Course_Price"])
y = df["Course_Price"]

# ✅ Save final feature names
feature_names = list(X.columns)
joblib.dump(feature_names, "feature_names.pkl")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Preprocessing pipeline
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numerical_features)
], remainder="passthrough")  # Keep one-hot encoded features

# ✅ Train model with hyperparameter tuning
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="r2", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ✅ Final model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", best_model)
])

pipeline.fit(X_train, y_train)

# ✅ Save model and preprocessor
joblib.dump(pipeline, "course_price_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model and Preprocessor saved successfully!")
