import os
import sys
from src.components.data_ingestion import DATA_INGESTION
from src.components.data_transformation import DATA_TRANSFORMATION
from src.components.model_trainer import MODEL_TRAINER
from src.logger import logging
from src.exception import CustomException

try:
    ingestion = DATA_INGESTION()
    _ ,train , test = ingestion.ingestion()

except Exception as e:
    raise CustomException(e,sys)

try:
    transform = DATA_TRANSFORMATION()
    train_arr, test_arr = transform.data_transformation_initiator(train, test)

except Exception as e:
    raise CustomException(e,sys)

try:
    trainer = MODEL_TRAINER()
    oof, test_pred, R2_score, RMSE, test_R2, test_RMSE= trainer.model_trainer_initiator(train_array= train_arr, test_array= test_arr)
except Exception as e:
    raise CustomException(e,sys)