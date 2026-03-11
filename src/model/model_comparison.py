import pandas as pd
import numpy as np
import os
import json
import logging
import yaml
import matplotlib

# Required for CI environments without display
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise RuntimeError(f"Error loading parameters from {params_path}: {e}")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame):
    try:
        X = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def plot_model_comparison(results: dict, save_path: str):

    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, model_name in enumerate(models):
        values = [results[model_name][m] for m in metrics]

        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=model_name,
            color=colors[i]
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                fontsize=9,
            )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)

    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logging.info(f"Model comparison chart saved to {save_path}")


def main():

    params_path = os.getenv("PARAMS_PATH", "params.yaml")

    train_data_path = os.getenv(
        "TRAIN_DATA_PATH",
        "./data/processed/train_processed_mean.csv",
    )

    test_data_path = os.getenv(
        "TEST_DATA_PATH",
        "./data/processed/test_processed_mean.csv",
    )

    comparison_metrics_path = os.getenv(
        "METRICS_PATH",
        "reports/model_comparison.json",
    )

    comparison_chart_path = os.getenv(
        "CHART_PATH",
        "reports/figures/model_comparison.png",
    )

    logging.info("Loading parameters")
    params = load_params(params_path)

    n_estimators = params["model_building"]["n_estimators"]

    logging.info("Loading datasets")

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Training Random Forest")

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
    )

    rf_model.fit(X_train, y_train)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    logging.info("Training SVM")

    svm_model = SVC(kernel="rbf", random_state=42)

    svm_model.fit(X_train_scaled, y_train)

    svm_metrics = evaluate_model(svm_model, X_test_scaled, y_test)

    logging.info("Training Logistic Regression")

    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    lr_model.fit(X_train_scaled, y_train)

    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test)

    results = {
        "Random Forest": rf_metrics,
        "SVM": svm_metrics,
        "Logistic Regression": lr_metrics,
    }

    best_model_name = max(results, key=lambda k: results[k]["accuracy"])

    logging.info("Best model: %s", best_model_name)

    os.makedirs(os.path.dirname(comparison_metrics_path), exist_ok=True)

    with open(comparison_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info("Metrics saved to %s", comparison_metrics_path)

    os.makedirs(os.path.dirname(comparison_chart_path), exist_ok=True)

    plot_model_comparison(results, comparison_chart_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Pipeline failed")
        raise
