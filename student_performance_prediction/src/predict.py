"""Predict student performance from user input."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import FEATURE_COLUMNS, preprocess_single_student
from src.evaluate_model import load_model_package


def get_float_input(prompt: str, minimum: float, maximum: float) -> float:
    """Read a float input and keep asking until the value is valid."""

    while True:
        try:
            value = float(input(prompt))
            if minimum <= value <= maximum:
                return value
            print(f"Please enter a value from {minimum} to {maximum}.")
        except ValueError:
            print("Please enter a valid number.")


def get_choice_input(prompt: str, valid_choices: list[str]) -> str:
    """Read a text input and keep asking until it matches an allowed choice."""

    normalized_choices = {choice.lower(): choice for choice in valid_choices}

    while True:
        value = input(prompt).strip().lower()
        if value in normalized_choices:
            return normalized_choices[value]
        print(f"Please choose one of: {', '.join(valid_choices)}")


def predict_student_performance() -> str:
    """Ask the user for student details and print the predicted performance."""

    model_package = load_model_package()
    model = model_package["model"]
    scaler = model_package["scaler"]
    encoders = model_package["encoders"]

    print("\nEnter Student Details")
    print("-" * 30)

    student_data = pd.DataFrame(
        [
            {
                "attendance_percentage": get_float_input("Attendance Percentage: ", 0, 100),
                "assignment_score": get_float_input("Assignment Score: ", 0, 100),
                "midterm_marks": get_float_input("Mid-Term Marks: ", 0, 100),
                "previous_grade": get_choice_input("Previous Grade (A/B/C/D): ", ["A", "B", "C", "D"]),
                "study_hours": get_float_input("Study Hours Per Day: ", 0, 12),
                "participation": get_choice_input("Participation (Yes/No): ", ["Yes", "No"]),
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prepared_student = preprocess_single_student(student_data, scaler, encoders)
    encoded_prediction = model.predict(prepared_student)
    predicted_label = encoders["final_performance"].inverse_transform(encoded_prediction)[0]

    print(f"\nPredicted Performance: {predicted_label}")
    return predicted_label


if __name__ == "__main__":
    predict_student_performance()
