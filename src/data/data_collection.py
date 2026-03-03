import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath : str) -> dict:
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")


#test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def load_data(filepath : str) -> pd.DataFrame :
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} :{e}")


def split_data(data : pd.DataFrame, test_size: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size= test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splittin data : {e}")

#train_data, test_data = train_test_split(data, test_size= test_size, random_state=42)

def save_data(df : pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} :{e}")
    

def main():
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data","raw")
# data_path = os.path.join("data","raw")
    try:
        params = load_params(params_filepath)
        test_size = params["data_collection"]["test_size"]
        data_filepath = params["data_collection"]["data_source"]
        
        data = load_data(data_filepath)
        train_data,test_data = split_data(data, test_size)

        os.makedirs(raw_data_path, exist_ok=True)

        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred :{e}")
    
if __name__ == "__main__":
    main()