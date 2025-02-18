import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
import pickle

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("model_trainer")
logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_trainer.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def model_trainer(X_trian_path:str,y_train_path:str):
    try:
        
        X_train = pd.read_csv(X_trian_path)
        y_train = pd.read_csv(y_train_path)

        rfc = RandomForestClassifier(
            n_estimators= 50,
            max_depth= 5,
            min_samples_split= 7
        )

        rfc.fit(X_train,y_train)
        logger.info("Model fitted to data successfully!!!%s")
        return rfc

    except Exception as e:
        print(e)
        logger.info("Unexcepected error -%s",e)

def save_model(model_obj,model_obj_path):

    try:
        model_dir_path = os.path.join(model_obj_path,"models")
        os.makedirs(model_dir_path,exist_ok=True)
        with open(os.path.join(model_dir_path,"rfc.pkl"),"wb") as f:
            pickle.dump(model_obj,f)
        logger.info("Model saved successfully!!!%s")
    except Exception as e:
        print(e) 
        logger.info("Unexcepected error -%s",e)

def main():
    try:
        X_train_path = "data/scaled_data/X_train_scl.csv"
        y_train = "data/labels/y_train.csv"
        rfc = model_trainer(X_train_path,y_train)
        save_model(rfc,"./model_objects")

    except Exception as e:
        print(e)
        logger.info("Unexcepected error -%s",e)

if __name__ == "__main__":
    main()