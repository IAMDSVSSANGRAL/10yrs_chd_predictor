import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging
from src.predictor.utils import save_object, read_csv_data

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def remove_outliers_iqr(self, df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def blood_pressure_classification(self, SysBP, DiaBP):
        if (SysBP < 90) and (DiaBP < 60): return 0
        elif (SysBP < 120) and (DiaBP < 80): return 1
        elif (SysBP < 130) and (DiaBP < 85): return 2
        elif (SysBP < 140) and (DiaBP < 90): return 3
        elif (SysBP >= 140) and (DiaBP < 90): return 4
        elif (SysBP < 160) and (DiaBP < 100): return 5
        elif (SysBP < 180) and (DiaBP < 110): return 6
        else: return 7

    def diabetes_grade(self, glucose):
        if glucose < 100: return 1
        elif glucose < 125: return 2
        elif glucose < 200: return 3
        elif glucose < 400: return 4
        else: return 5

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = read_csv_data(train_path)
            test_df = read_csv_data(test_path)

            logging.info("Read train and test data")

            # drop id
            train_df.drop('id', axis=1, inplace=True)
            test_df.drop('id', axis=1, inplace=True)

            # handle missing values
            impute_cols = ['education', 'cigsperday', 'bpmeds', 'totchol', 'bmi', 'heartrate', 'glucose']
            for col in impute_cols:
                imputer = SimpleImputer(strategy='median')
                train_df[col] = imputer.fit_transform(train_df[[col]])
                test_df[col] = imputer.transform(test_df[[col]])

            logging.info("Missing values handled")

            # remove outliers
            outlier_cols = ['age', 'totchol', 'sysbp', 'diabp', 'bmi', 'heartrate', 'glucose']
            train_df = self.remove_outliers_iqr(train_df, outlier_cols)

            logging.info("Outliers removed from training set")

            # new features
            train_df['hypertension'] = train_df.apply(lambda x: self.blood_pressure_classification(x['sysbp'], x['diabp']), axis=1)
            test_df['hypertension'] = test_df.apply(lambda x: self.blood_pressure_classification(x['sysbp'], x['diabp']), axis=1)

            train_df['Diabetes_grade'] = train_df['glucose'].apply(lambda x: self.diabetes_grade(x))
            test_df['Diabetes_grade'] = test_df['glucose'].apply(lambda x: self.diabetes_grade(x))

            train_df['mean_art_pressure'] = (train_df['sysbp'] + 2 * train_df['diabp']) / 3
            test_df['mean_art_pressure'] = (test_df['sysbp'] + 2 * test_df['diabp']) / 3

            logging.info("New features created")

            target_column = 'tenyearchd'
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Column categorization
            num_features = ['age', 'totchol', 'sysbp', 'diabp', 'bmi', 'heartrate', 'glucose',
                            'mean_art_pressure']
            ord_features = ['education']
            nom_features = ['sex', 'is_smoking']
            bin_features = ['bpmeds', 'prevalentstroke', 'prevalenthyp', 'diabetes']
            new_cat_features = ['hypertension', 'Diabetes_grade']

            # Pipelines
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            ord_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_features),
                ('ord', ord_pipeline, ord_features),
                ('nom', cat_pipeline, nom_features),
                ('bin', 'passthrough', bin_features),
                ('new_cat', 'passthrough', new_cat_features)
            ])

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)
            logging.info("Preprocessor saved")

            return (X_train_processed,  X_test_processed, self.transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise TenYearChdException(e, sys)
