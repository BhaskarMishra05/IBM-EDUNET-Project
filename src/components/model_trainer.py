import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class MODEL_TRAINER_CONFIG:
    model_path: str = os.path.join("artifacts", "model.pkl")
    kmeans_path: str = os.path.join("artifacts", "kmeans.pkl")


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


            kmeans = KMeans(n_clusters=10, random_state=42)
            train_clusters = kmeans.fit_predict(y_train.reshape(-1, 1))
            test_clusters = kmeans.predict(y_test.reshape(-1, 1))

            save_obj(self.model_trainer_config.kmeans_path, kmeans)

            models = {}
            oof_pred = np.zeros(len(X_train))
            test_pred = np.zeros(len(X_test))

            for cluster in range(10):
                logging.info(f"Training cluster {cluster} model...")

                cluster_indices = np.where(train_clusters == cluster)[0]
                X_cluster = X_train[cluster_indices]
                y_cluster = y_train[cluster_indices]

                test_cluster_indices = np.where(test_clusters == cluster)[0]
                X_test_cluster = X_test[test_cluster_indices]

                cluster_oof = np.zeros(len(X_cluster))
                cluster_test_pred = np.zeros(len(X_test_cluster))

                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster, y_cluster)):
                    X_tr, X_val = X_cluster[train_idx], X_cluster[val_idx]
                    y_tr, y_val = y_cluster[train_idx], y_cluster[val_idx]

                    model = LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=7,
                        num_leaves=31,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    )
                    model.fit(X_tr, y_tr)


                    cluster_oof[val_idx] = model.predict(X_val)

                    cluster_test_pred += model.predict(X_test_cluster) / kf.n_splits

                oof_pred[cluster_indices] = cluster_oof
                test_pred[test_cluster_indices] = cluster_test_pred

                models[cluster] = model

            import matplotlib.pyplot as plt

            r2 = r2_score(y_train, oof_pred)
            rmse = np.sqrt(mean_squared_error(y_train, oof_pred))
            logging.info(f"OOF R2: {r2:.4f}, RMSE: {rmse:.4f}")


            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            logging.info(f"TEST R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

            save_obj(self.model_trainer_config.model_path, models)


            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(y_train, oof_pred, alpha=0.5, edgecolor='k')
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
            plt.title(f"Train: R2={r2:.2f}, RMSE={rmse:.0f}")
            plt.xlabel("Actual Salary")
            plt.ylabel("Predicted Salary")
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, test_pred, alpha=0.5, edgecolor='k')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f"Test: R2={test_r2:.2f}, RMSE={test_rmse:.0f}")
            plt.xlabel("Actual Salary")
            plt.ylabel("Predicted Salary")

            plt.tight_layout()
            plot_path = os.path.join("artifacts", "predicted_vs_actual.png")
            plt.savefig(plot_path)
            plt.close()

            logging.info(f"Prediction plot saved at: {plot_path}")
            print("Y_train stats → min:", y_train.min(), "max:", y_train.max())
            print("Y_test stats → min:", y_test.min(), "max:", y_test.max())


            return oof_pred, test_pred, r2, rmse, test_r2, test_rmse

        except Exception as e:
            raise CustomException(e, sys)
