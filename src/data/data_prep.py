import os
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception:
        logging.exception(f"Error loading data from {filepath}")
        raise


def fill_missing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    try:
        imputer_dict = {}
        for column in train_df.columns:
            if train_df[column].isnull().any() or test_df[column].isnull().any():
                # Compute mean strictly on train data to prevent data leakage
                mean_value = train_df[column].mean()
                imputer_dict[column] = mean_value

                # Apply computed mean to both sets
                train_df[column].fillna(mean_value, inplace=True)
                test_df[column].fillna(mean_value, inplace=True)

        return train_df, test_df, imputer_dict
    except Exception:
        logging.exception("Error filling missing values")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception:
        logging.exception(f"Error saving data to {filepath}")
        raise


def main():
    raw_data_path = "./data/raw/"
    processed_data_path = "./data/processed"
    model_path = "./models"

    try:
        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        train_processed_data, test_processed_data, imputer_dict = fill_missing(train_data, test_data)

        os.makedirs(processed_data_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Save imputer dictionary
        with open(os.path.join(model_path, "imputer.pkl"), "wb") as f:
            pickle.dump(imputer_dict, f)

        save_data(
            train_processed_data,
            os.path.join(processed_data_path, "train_processed_mean.csv"),
        )
        save_data(
            test_processed_data,
            os.path.join(processed_data_path, "test_processed_mean.csv"),
        )
        logging.info("Data preparation completed successfully")

    except Exception:
        logging.exception("An error occurred during data preparation")
        raise


if __name__ == "__main__":
    main()
