import os
import sys
from src.utils import load_obj
import pandas as pd
import numpy as np
class MAIN:
    def __init__(self):
        pass
    def predict(self, features):
        model_path = 'artifacts/model.pkl'
        preprocessing_path = 'artifacts/preprocessed.pkl'

        load_model = load_obj(model_path)
        load_preprocesssing = load_obj(preprocessing_path)
        if isinstance(features, np.ndarray):
            raise ValueError("Pass features as DataFrame with correct column names")

        data_load = load_preprocesssing.transform(features)
        pred= load_model.predict(data_load)
        return pred
class DATA:
    def __init__(self,age, workclass, fnlwgt, education, educational_num,
       marital_status, occupation, relationship, race, gender,
       capital_gain, capital_loss, hours_per_week, native_country):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.educational_num = educational_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.gender = gender
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def data_frame(self):
        data_dict = {
            'age': [self.age],
            'workclass': [self.workclass],
            'fnlwgt': [self.fnlwgt],
            'education': [self.education],
            'educational_num': [self.educational_num],
            'marital_status': [self.marital_status],
            'occupation': [self.occupation],
            'relationship': [self.relationship],
            'race': [self.race],
            'gender': [self.gender],
            'capital_gain': [self.capital_gain],
            'capital_loss': [self.capital_loss],
            'hours_per_week': [self.hours_per_week],
            'native_country': [self.native_country]
        }
        return pd.DataFrame(data_dict)