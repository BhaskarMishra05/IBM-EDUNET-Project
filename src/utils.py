import os
import sys
from src.logger import logging
from src.exception import CustomException
import joblib
def save_obj(path: str, obj):
    with open (path,'wb') as path_obj:
        joblib.dump(obj , path_obj)
def load_obj(path: str):
    with open (path, 'rb') as path_obj:
        joblib.load(path_obj)