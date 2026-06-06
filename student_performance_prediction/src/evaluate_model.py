"""Evaluate the saved student performance model."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import FEATURE_COLUMNS
from src.train_model import DATA_PATH, MODEL_PATH, train_model


OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_model_package(model_path: str | Path = MODEL_PATH) -> dict:
    """Load the saved pickle model package."""

    model_path = Path(model_path)
    if not model_path.exists():
        print("Saved model not found. Training a model first...")
        return train_model()

    with model_path.open("rb") as file:
        return pickle.load(file)


def plot_confusion_matrix(y_true, y_pred, class_names) -> None:
    """Save a confusion matrix chart."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_feature_importance(model_package: dict) -> None:
    """Save a feature importance chart for Random Forest models."""

    model = model_package.get("trained_models", {}).get("Random Forest")
    if not hasattr(model, "feature_importances_"):
        print("Random Forest feature importance is not available.")
        return

    importance_data = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=importance_data,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print("\nFeature Importance")
    print("-" * 30)
    for _, row in importance_data.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

    top_feature = importance_data.iloc[0]["feature"]
    print(f"\nMost influential factor in this model: {top_feature}")


def evaluate_model() -> None:
    """Print evaluation metrics and save evaluation charts."""

    model_package = load_model_package()
    model = model_package["model"]
    x_test = model_package["x_test"]
    y_test = model_package["y_test"]
    target_encoder = model_package["encoders"]["final_performance"]
    class_names = target_encoder.classes_

    predictions = model.predict(x_test)

    print("\nEvaluation Results")
    print("-" * 30)
    print(f"Model: {model_package['model_name']}")
    print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    plot_confusion_matrix(y_test, predictions, class_names)
    plot_feature_importance(model_package)

    print(f"\nCharts saved inside: {OUTPUTS_DIR}")
    print(f"Dataset used: {DATA_PATH}")


if __name__ == "__main__":
    evaluate_model()
