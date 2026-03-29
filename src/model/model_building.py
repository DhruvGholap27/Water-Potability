import os
import pickle
import pandas as pd
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(params_path: str) -> tuple[int, int]:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"], params["base"]["random_state"]
    except Exception:
        logging.exception(f"Error loading parameters from {params_path}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception:
        logging.exception(f"Error loading data from {filepath}")
        raise


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return X, y
    except Exception:
        logging.exception("Error preparing data")
        raise


def train_model(
    X: pd.DataFrame, y: pd.Series, n_estimators: int, random_state: int
) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X, y)
        return clf
    except Exception:
        logging.exception("Error training model")
        raise


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception:
        logging.exception(f"Error saving model to {filepath}")
        raise


def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed_mean.csv"
        model_name = "models/model.pkl"

        n_estimators, random_state = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators, random_state)
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        save_model(model, model_name)
        logging.info("Model training completed successfully")

    except Exception:
        logging.exception("An error occurred during model building")
        raise


if __name__ == "__main__":
    main()
