import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DATA_INGESTION_CONFIG:
    raw_csv_path: str = os.path.join('artifacts','raw.csv')
    train_csv_path: str = os.path.join('artifacts','train.csv')
    test_csv_path: str = os.path.join('artifacts','test.csv')

class DATA_INGESTION:
    def __init__(self):
        self.data_ingestion_conifg = DATA_INGESTION_CONFIG()

    def ingestion(self):
        try:
            logging.info('Making a directory for storing files')
            os.makedirs(os.path.dirname(self.data_ingestion_conifg.raw_csv_path), exist_ok=True)
            logging.info('loading raw csv file')
            df = pd.read_csv(self.data_ingestion_conifg.raw_csv_path)
            logging.info('Successfully loaded the dataset')
            logging.info('Splitting the dataset into train and test datasets')
            train_datasets, test_dataset = train_test_split(df, random_state= 42, test_size=0.25)
            logging.info('Successfully splitted the dataset')
            logging.info('Converting the datasets back into individual csv files')
            train_datasets.to_csv(self.data_ingestion_conifg.train_csv_path, index=False)
            test_dataset.to_csv(self.data_ingestion_conifg.test_csv_path, index= False)
            logging.info('Data Ingestion Completed Successfully')
            return (
                self.data_ingestion_conifg.raw_csv_path,
                self.data_ingestion_conifg.train_csv_path,
                self.data_ingestion_conifg.test_csv_path
            )

            
        except Exception as e:
            raise CustomException (e,sys)

