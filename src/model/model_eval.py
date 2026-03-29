import os
import pandas as pd
import json
import pickle
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception:
        logging.exception(f"Error loading data from {filepath}")
        raise


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split data into features (X) and target (y)"""
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception:
        logging.exception("Error preparing data")
        raise


def load_model(filepath: str):
    """Load a pickled model from file"""
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception:
        logging.exception(f"Error loading model from {filepath}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance and return metrics dictionary"""
    try:
        y_pred = model.predict(X_test)

        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        return metrics_dict
    except Exception:
        logging.exception("Error evaluating model")
        raise


def save_metrics(metrics: dict, metrics_path: str) -> None:
    """Save evaluation metrics as a JSON file"""
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception:
        logging.exception(f"Error saving metrics to {metrics_path}")
        raise


def main():
    try:
        test_data_path = "./data/processed/test_processed_mean.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)

        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        save_metrics(metrics, metrics_path)

        logging.info(f"Metrics successfully saved to {metrics_path}")

    except Exception:
        logging.exception("An error occurred during model evaluation")
        raise


if __name__ == "__main__":
    main()
