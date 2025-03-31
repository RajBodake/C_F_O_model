import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load trained model and preprocessor
model_path = "course_price_model.pkl"
preprocessor_path = "preprocessor.pkl"
feature_names_path = "feature_names.pkl"

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path) or not os.path.exists(feature_names_path):
    st.error("❌ Model or preprocessor file not found. Please train the model first!")
    st.stop()

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
feature_names = joblib.load(feature_names_path)  # Correct feature order

# ✅ Load dataset for visualization
file_path = r"D:\C_F_O.model\course_data_fixed.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    df = None

# ✅ Streamlit UI
st.title("📘 Course Price Prediction App")
st.write("Enter course details to predict the price.")

# ✅ Input fields
course_duration = st.number_input("Course Duration (hours)", min_value=1, value=10)
num_modules = st.number_input("Number of Modules", min_value=1, value=5)
certification = st.selectbox("Certification", ["Yes", "No"])
competitor_price = st.number_input("Competitor Price", min_value=0, value=1000)
student_demand = st.number_input("Student Demand (1-1000)", min_value=1, max_value=1000, value=50)
dropout_rate = st.slider("Dropout Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
feedback_score = st.slider("Feedback Score (1-5)", min_value=1.0, max_value=5.0, value=4.0)
marketing_spend = st.number_input("Marketing Spend (₹)", min_value=0, value=5000)
discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
difficulty_level = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

# ✅ Convert categorical inputs
certification = 1 if certification == "Yes" else 0

# ✅ Create DataFrame for input
input_df = pd.DataFrame([[course_duration, num_modules, certification, competitor_price, 
                          student_demand, dropout_rate, feedback_score, marketing_spend, 
                          discount_offered, difficulty_level]],
                        columns=["Course_Duration", "Num_Modules", "Certification", "Competitor_Price",
                                 "Student_Demand", "Dropout_Rate", "Feedback_Score", "Marketing_Spend", 
                                 "Discount_Offered", "Difficulty_Level"])

# ✅ Ensure one-hot encoding for Difficulty_Level
difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
for level in difficulty_levels:
    input_df[f"Difficulty_Level_{level}"] = (input_df["Difficulty_Level"] == level).astype(int)

# ✅ Drop original `Difficulty_Level` column
input_df.drop(columns=["Difficulty_Level"], inplace=True)

# ✅ Reorder columns to match model training
missing_features = set(feature_names) - set(input_df.columns)
extra_features = set(input_df.columns) - set(feature_names)

if missing_features:
    st.error(f"❌ Error making prediction: Missing features {missing_features}")
    st.stop()
if extra_features:
    st.error(f"❌ Error making prediction: Unexpected extra features {extra_features}")
    st.stop()

input_df = input_df[feature_names]

# ✅ Prediction button
if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Course Price: ₹{prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

# ✅ Data Visualizations (if dataset is available)
if df is not None:
    st.subheader("📊 Course Price Insights")

    # ✅ Price Distribution
    fig, ax = plt.subplots()
    sns.histplot(df["Course_Price"], bins=30, kde=True, ax=ax, color='blue')
    ax.set_title("Course Price Distribution")
    st.pyplot(fig)

    # ✅ Price vs Student Demand
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Student_Demand"], y=df["Course_Price"], alpha=0.6, ax=ax)
    ax.set_title("Price vs Student Demand")
    st.pyplot(fig)

    # ✅ Price by Difficulty Level
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Difficulty_Level"], y=df["Course_Price"], ax=ax)
    ax.set_title("Price by Difficulty Level")
    st.pyplot(fig)

else:
    st.warning("⚠ No dataset found for visualization.")
