"""Data generation and preprocessing helpers.

This file keeps all data-related work in one place:
- create a realistic synthetic student dataset
- load the CSV file
- clean missing values
- encode categorical text values
- split the data into train and test sets
- scale numerical features
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLUMNS = [
    "attendance_percentage",
    "assignment_score",
    "midterm_marks",
    "previous_grade",
    "study_hours",
    "participation",
]

TARGET_COLUMN = "final_performance"


def generate_synthetic_data(
    output_path: str | Path,
    number_of_records: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create and save a realistic synthetic student performance dataset.

    The target label is not random. It is based on a weighted academic score
    that uses attendance, assignments, mid-term marks, previous grade,
    study hours, and activity participation.
    """

    rng = np.random.default_rng(random_state)

    attendance = np.clip(rng.normal(loc=78, scale=12, size=number_of_records), 35, 100)
    assignment = np.clip(rng.normal(loc=72, scale=15, size=number_of_records), 20, 100)
    midterm = np.clip(rng.normal(loc=68, scale=16, size=number_of_records), 15, 100)
    study_hours = np.clip(rng.gamma(shape=2.2, scale=1.4, size=number_of_records), 0.25, 9)

    previous_grades = rng.choice(
        ["A", "B", "C", "D"],
        size=number_of_records,
        p=[0.22, 0.36, 0.29, 0.13],
    )

    participation = rng.choice(
        ["Yes", "No"],
        size=number_of_records,
        p=[0.62, 0.38],
    )

    grade_points = {"A": 92, "B": 78, "C": 62, "D": 45}
    participation_bonus = np.where(participation == "Yes", 6, 0)
    previous_grade_score = np.array([grade_points[grade] for grade in previous_grades])

    academic_score = (
        attendance * 0.20
        + assignment * 0.20
        + midterm * 0.25
        + previous_grade_score * 0.20
        + study_hours * 4.0
        + participation_bonus
        + rng.normal(loc=0, scale=5, size=number_of_records)
    )

    final_performance = np.select(
        [academic_score >= 78, academic_score >= 58],
        ["Good", "Average"],
        default="Poor",
    )

    data = pd.DataFrame(
        {
            "student_id": np.arange(1, number_of_records + 1),
            "attendance_percentage": attendance.round(2),
            "assignment_score": assignment.round(2),
            "midterm_marks": midterm.round(2),
            "previous_grade": previous_grades,
            "study_hours": study_hours.round(2),
            "participation": participation,
            "final_performance": final_performance,
        }
    )

    # Add a tiny amount of missing data so beginners can learn how it is handled.
    for column in FEATURE_COLUMNS:
        missing_indexes = rng.choice(
            data.index,
            size=max(1, int(number_of_records * 0.01)),
            replace=False,
        )
        data.loc[missing_indexes, column] = np.nan

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the student dataset from a CSV file."""

    return pd.read_csv(csv_path)


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple beginner-friendly strategies."""

    cleaned_data = data.copy()

    # Numerical columns are filled with their median because the median is
    # less affected by unusually high or low values.
    numeric_columns = cleaned_data.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())

    # Categorical columns are filled with their most common value.
    categorical_columns = cleaned_data.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mode()[0])

    return cleaned_data


def encode_categorical_columns(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Convert text columns into numeric values using LabelEncoder."""

    encoded_data = data.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for column in ["previous_grade", "participation", TARGET_COLUMN]:
        encoder = LabelEncoder()
        encoded_data[column] = encoder.fit_transform(encoded_data[column])
        encoders[column] = encoder

    return encoded_data, encoders


def prepare_train_test_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, Dict[str, LabelEncoder]]:
    """Prepare scaled train and test arrays for machine learning models."""

    cleaned_data = handle_missing_values(data)
    encoded_data, encoders = encode_categorical_columns(cleaned_data)

    x = encoded_data[FEATURE_COLUMNS]
    y = encoded_data[TARGET_COLUMN]

    # Stratify keeps the class proportions similar in train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, encoders


def preprocess_single_student(
    student_data: pd.DataFrame,
    scaler: StandardScaler,
    encoders: Dict[str, LabelEncoder],
) -> np.ndarray:
    """Prepare one student record for prediction."""

    prepared_data = student_data.copy()
    prepared_data["previous_grade"] = encoders["previous_grade"].transform(
        prepared_data["previous_grade"]
    )
    prepared_data["participation"] = encoders["participation"].transform(
        prepared_data["participation"]
    )

    return scaler.transform(prepared_data[FEATURE_COLUMNS])
