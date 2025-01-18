import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("data_ingention")
logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"Data_ingention.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def load_data(data_path:str)-> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info("Data Loaded successful...")
        return df
    except Exception as e:
        logger.error("Unexpected error occured...",e)
        print(e)

def preprocessing_data(data:pd.DataFrame):
    try:
        df = data.copy()
        df = df.drop(['customer_id','country',],axis = 1)
        logger.info("Data preprocessing complete...")
        return df
    except Exception as e:
        print(e)
        logger.error("Unexpected error occured...",e)

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,raw_data_path:str):
    try:
        
        raw_data_path = os.path.join(raw_data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train_data.csv"),index=True)
        test_data.to_csv(os.path.join(raw_data_path,"test_data.csv"),index=True)
        logger.info("Train data and test data saved successful...")

    except Exception as e:
        print(e)
        logger.error("Unexpected error occured...",e)

def main(data_path:str):
    try:
        test_size = 0.3
        
        df = load_data(data_path)
        final_df = preprocessing_data(df)

        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)

        save_data(train_data,test_data,"./data")

    except Exception as e:
        print(e)
        logger.error("Unexpected error occured...",e)

if __name__ == "__main__":
    main("experiments/Bank Customer Churn Prediction.csv")
    




