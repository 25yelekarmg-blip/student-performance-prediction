import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import FEATURE_COLUMNS, preprocess_single_student
from src.evaluate_model import load_model_package


st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("Student Performance Prediction System")

st.write("Enter student details to predict final performance.")

attendance = st.slider("Attendance Percentage", 0, 100, 75)
assignment = st.slider("Assignment Score", 0, 100, 70)
midterm = st.slider("Mid-Term Marks", 0, 100, 65)
previous_grade = st.selectbox("Previous Grade", ["A", "B", "C", "D"])
study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 3.0)
participation = st.selectbox("Participation", ["Yes", "No"])

if st.button("Predict Performance"):
    model_package = load_model_package()

    model = model_package["model"]
    scaler = model_package["scaler"]
    encoders = model_package["encoders"]

    student_data = pd.DataFrame(
        [
            {
                "attendance_percentage": attendance,
                "assignment_score": assignment,
                "midterm_marks": midterm,
                "previous_grade": previous_grade,
                "study_hours": study_hours,
                "participation": participation,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prepared_student = preprocess_single_student(student_data, scaler, encoders)
    encoded_prediction = model.predict(prepared_student)
    predicted_label = encoders["final_performance"].inverse_transform(encoded_prediction)[0]

    st.success(f"Predicted Performance: {predicted_label}")