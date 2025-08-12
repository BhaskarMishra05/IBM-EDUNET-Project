import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class MODEL_TRAINER_CONFIG:
    model_path: str = os.path.join("artifacts", "model.pkl")
    


class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config = MODEL_TRAINER_CONFIG()

    def model_trainer_initiator(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Model training initiated with clustering")

            target_index = -1  
            X_train = train_array[:, :target_index]
            y_train = train_array[:, target_index]

            X_test = test_array[:, :target_index]
            y_test = test_array[:, target_index]


            baseline_model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)
            
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            baseline_precision = precision_score(y_test, baseline_pred, average='weighted')
            baseline_recall = recall_score(y_test, baseline_pred, average='weighted')
            baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
            cm = confusion_matrix(y_test, baseline_pred)
            logging.info(f"Trainig Accuracy : {baseline_model.score(X_train, y_train)}")
            logging.info(f"Testing Accuracy : {baseline_model.score(X_test, y_test)}")
            logging.info(f"Baseline Accuracy : {baseline_accuracy:.4f}")
            logging.info(f"Baseline Precision: {baseline_precision:.4f}")
            logging.info(f"Baseline Recall   : {baseline_recall:.4f}")
            logging.info(f"Baseline F1-score : {baseline_f1:.4f}")
            logging.info(f"Confusion Matrix:\n{cm}")

            save_obj(self.model_trainer_config.model_path, baseline_model)

            return baseline_accuracy
        except Exception as e:
            raise CustomException(e, sys)
