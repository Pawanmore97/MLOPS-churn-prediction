import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.compose import ColumnTransformer

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def feature_encoding(train_path:str,test_path:str):

    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        train_data = pd.get_dummies(train_data,drop_first=True)
        test_data = pd.get_dummies(test_data,drop_first=True)
        logger.info("Feature encoding done!!!%s")
        return train_data,test_data

    except Exception as e:
        print(e)
        logger.error("Unexpected error occured -%s",e)

def scale_data(train_data:pd.DataFrame,test_data:pd.DataFrame,raw_data_path:str):
    try:
        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1]
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1]

        transformer = ColumnTransformer(transformers=[
            ("balance",StandardScaler(),['balance']),
            ("estimated_salary",StandardScaler(),['estimated_salary'])
        ],remainder="passthrough")

        X_train_scl = transformer.fit_transform(X_train)
        X_test_scl = transformer.transform(X_test)

        logger.info("Feature scaling is done!!!%s")

        scale_data_dir_path = os.path.join(raw_data_path,"scaled_data")
        labels_dir_path = os.path.join(raw_data_path,"labels")

        os.makedirs(scale_data_dir_path,exist_ok=True)
        os.makedirs(labels_dir_path,exist_ok=True)

        X_train_scl = pd.DataFrame(X_train_scl,columns=X_train.columns)
        X_test_scl = pd.DataFrame(X_test_scl,columns=X_test.columns)


        X_train_scl.to_csv(os.path.join(scale_data_dir_path,"X_train_scl.csv"),index = False)
        X_test_scl.to_csv(os.path.join(scale_data_dir_path,"X_test_scl.csv"),index = False)
        y_train.to_csv(os.path.join(labels_dir_path,"y_train.csv"),index = False)
        y_test.to_csv(os.path.join(labels_dir_path,"y_test.csv"),index = False)


        logger.info("scaled files saved successfully!!!%s")
    except Exception as e:
        print(e)
        logger.error("Unexpected error occured -%s",e)

def main():
    try:

        train_path = os.path.join(os.getcwd(),"data","raw","train_data.csv")
        test_path = os.path.join(os.getcwd(),"data","raw","test_data.csv")

        train_data,test_data = feature_encoding(train_path,test_path)

        scale_data(train_data,test_data,"./data")

    except Exception as e:
        print(e)
        logger.error("Unexpected error occured -%s",e)

if __name__ == "__main__":
    main()
