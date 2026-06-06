"""Train and compare machine learning models."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import (
    generate_synthetic_data,
    load_dataset,
    prepare_train_test_data,
)


DATA_PATH = PROJECT_ROOT / "data" / "student_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_model.pkl"


def get_models() -> Dict[str, object]:
    """Return the models that will be trained and compared."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }


def train_and_compare_models(data: pd.DataFrame) -> Tuple[object, Dict[str, float], dict]:
    """Train all models and return the best model with useful artifacts."""

    x_train, x_test, y_train, y_test, scaler, encoders = prepare_train_test_data(data)
    models = get_models()
    accuracy_results: Dict[str, float] = {}
    trained_models: Dict[str, object] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy_results[model_name] = accuracy_score(y_test, predictions)
        trained_models[model_name] = model

    best_model_name = max(accuracy_results, key=accuracy_results.get)
    best_model = trained_models[best_model_name]

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "trained_models": trained_models,
        "scaler": scaler,
        "encoders": encoders,
        "accuracy_results": accuracy_results,
        "x_test": x_test,
        "y_test": y_test,
    }

    return best_model, accuracy_results, model_package


def save_model(model_package: dict, model_path: str | Path = MODEL_PATH) -> None:
    """Save the best model package using pickle."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(model_package, file)


def train_model() -> dict:
    """Generate/load data, train models, save the best model, and print results."""

    if not DATA_PATH.exists():
        print("Dataset not found. Generating synthetic student data...")
        generate_synthetic_data(DATA_PATH, number_of_records=1000)

    data = load_dataset(DATA_PATH)
    _, accuracy_results, model_package = train_and_compare_models(data)
    save_model(model_package)

    print("\nModel Accuracy Comparison")
    print("-" * 30)
    for model_name, accuracy in accuracy_results.items():
        print(f"{model_name}: {accuracy:.2%}")

    print(f"\nBest Model: {model_package['model_name']}")
    print(f"Model saved at: {MODEL_PATH}")
    return model_package


if __name__ == "__main__":
    train_model()
