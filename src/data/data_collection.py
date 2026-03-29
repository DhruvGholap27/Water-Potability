import os
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(filepath: str) -> dict:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params
    except Exception:
        logging.exception(f"Error loading parameters from {filepath}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception:
        logging.exception(f"Error loading data from {filepath}")
        raise


def split_data(
    data: pd.DataFrame,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
    except Exception:
        logging.exception("Error splitting data")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception:
        logging.exception(f"Error saving data to {filepath}")
        raise


def main():
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")

    try:
        params = load_params(params_filepath)
        test_size = params["data_collection"]["test_size"]
        data_filepath = params["data_collection"]["data_source"]
        random_state = params["base"]["random_state"]

        data = load_data(data_filepath)
        train_data, test_data = split_data(data, test_size, random_state)

        os.makedirs(raw_data_path, exist_ok=True)

        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
        logging.info("Data collection and splitting completed successfully")

    except Exception:
        logging.exception("An error occurred during data collection")
        raise


if __name__ == "__main__":
    main()
