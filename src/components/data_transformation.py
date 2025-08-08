import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

@dataclass
class DATA_TRANSFORMATION_COMFIG:
    preprocessing_path: str = os.path.join('artifacts', 'preprocessed.pkl')
 
class DATA_TRANSFORMATION:
    def __init__(self):
        self.data_transformation_config = DATA_TRANSFORMATION_COMFIG()

    def feature_engineering(self, df):
        try:
            df['exchange_rate']=  (df['salary']/df['salary_in_usd']).round(2)
            df['experience_level'] = df['experience_level'].map({'EN':0,
                                                                'MI':1,
                                                                'SE':2,
                                                                'EX':3})
            df['employment_type'] = df['employment_type'].map({'FL':0,
                                                                'PT':1,
                                                                'CT':2,
                                                                'FT':3})
            df['company_size'] = df['company_size'].map({'S':0,
                                                        'M':1,
                                                        'L':2})
            outlier_comps = ['salary_in_usd', 'employment_type']
            for i in outlier_comps:
                Q1=df[i].quantile(0.25)
                Q3=df[i].quantile(0.75)
                IQR=Q3-Q1
                lower_bound= Q1 - 1.5 * IQR
                upper_bound= Q3 + 1.5 * IQR
                df=df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

            df['is_location_same']= (df['employee_residence']== df['company_location']).astype(int)
            df = df.drop('salary', axis=1)
            df = df[[col for col in df.columns if col != 'salary_in_usd'] + ['salary_in_usd']]
            return df
        except Exception as e:
            raise CustomException (e, sys)

    def preprocessing(self, df: pd.DataFrame):
        try:

            '''df = pd.read_csv(df)'''

            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if 'salary_in_usd' in numerical_columns:
                numerical_columns.remove('salary_in_usd')

            numerical_pipeline= Pipeline([('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())])
            
            categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('encoder', OneHotEncoder(drop='first', handle_unknown= 'ignore', sparse_output= False))])
            

            preprocessing = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_columns),
                                            ('categorical_pipeline', categorical_pipeline, categorical_columns)])
            
            return preprocessing
        
        except Exception as e:
            raise CustomException (e, sys)
    def data_transformation_initiator(self, train_file , test_file):
        train_raw = pd.read_csv(train_file)
        test_raw = pd.read_csv(test_file)

        train_feature_engg = self.feature_engineering(train_raw)
        test_feature_engg = self.feature_engineering(test_raw)

        train_feature = train_feature_engg.drop(columns=['salary_in_usd'], axis= 1)
        train_target = train_feature_engg['salary_in_usd']
        test_feature = test_feature_engg.drop(columns=['salary_in_usd'], axis= 1)
        test_target = test_feature_engg['salary_in_usd']

        preprocessing_obj = self.preprocessing(train_feature)

        train_feature_preprocessed = preprocessing_obj.fit_transform(train_feature)
        test_feature_preprocessed = preprocessing_obj.transform(test_feature)

        train_arr = np.c_[train_feature_preprocessed, train_target]
        test_arr = np.c_[test_feature_preprocessed, test_target]
        logging.info(f"Y_train stats → min: {train_target.min()} max: {train_target.max()}")
        logging.info(f"Y_test stats → min: {test_target.min()} max: {test_target.max()}")

        save_obj(self.data_transformation_config.preprocessing_path, preprocessing_obj)

        return (
            train_arr, test_arr
        )
