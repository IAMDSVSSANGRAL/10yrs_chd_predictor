import os
import sys
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

import joblib
from dataclasses import dataclass

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging
from src.predictor.utils import save_object, read_csv_data


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, use_smoteenn=False):
        self.transformation_config = DataTransformationConfig()
        self.use_smoteenn = use_smoteenn

    # ---------- Outlier Capping ----------
    def cap_outliers_iqr(self, df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.25 * IQR
            upper = Q3 + 1.25 * IQR
            df[col] = np.where(df[col] < lower, lower,
                               np.where(df[col] > upper, upper, df[col]))
        return df

    # ---------- Feature Engineering ----------
    def blood_pressure_classification(self, SysBP, DiaBP):
        if (SysBP < 90) and (DiaBP < 60): return 0
        if (SysBP < 120) and (DiaBP < 80): return 1
        if (SysBP < 130) and (DiaBP < 86): return 2
        if (SysBP < 140) and (DiaBP < 90): return 3
        if (SysBP > 140) and (DiaBP < 90): return 4
        if (SysBP < 160) and (DiaBP < 100): return 5
        if (SysBP < 180) and (DiaBP < 110): return 6
        return 7

    def diabetes_grade(self, glucose):
        if glucose < 100: return 1
        if glucose < 125: return 2
        if glucose < 200: return 3
        if glucose < 400: return 4
        return 5

    # ---------- Main Transformation ----------
    def initiate_data_transformation(self, train_path, test_path) -> tuple:
        try:
            train_df = read_csv_data(train_path)
            test_df = read_csv_data(test_path)

            logging.info("Read train and test data")

            # Drop ID column
            for df in [train_df, test_df]:
                if "id" in df.columns:
                    df.drop("id", axis=1, inplace=True)

            # ---------------- Handle Missing Values ----------------
            # Categorical
            train_df['education'].fillna(train_df['education'].mode()[0], inplace=True)
            test_df['education'].fillna(test_df['education'].mode()[0], inplace=True)

            train_df['bpmeds'].fillna(train_df['bpmeds'].mode()[0], inplace=True)
            test_df['bpmeds'].fillna(test_df['bpmeds'].mode()[0], inplace=True)

            # Numerical - mean imputation
            mean_cols = ['cigsperday', 'totchol', 'bmi', 'heartrate']
            for col in mean_cols:
                train_df[col].fillna(train_df[col].mean(), inplace=True)
                test_df[col].fillna(test_df[col].mean(), inplace=True)

            # Glucose - KNN imputer
            imputer = KNNImputer(n_neighbors=5)
            train_df[['glucose']] = imputer.fit_transform(train_df[['glucose']])
            test_df[['glucose']] = imputer.transform(test_df[['glucose']])

            logging.info("Missing values handled with mixed strategies")

            # ---------------- Outlier Capping ----------------
            num_outlier_cols = ['age', 'cigsperday', 'totchol', 'sysbp', 'diabp',
                                'bmi', 'heartrate', 'glucose']
            train_df = self.cap_outliers_iqr(train_df, num_outlier_cols)
            test_df = self.cap_outliers_iqr(test_df, num_outlier_cols)

            logging.info("Outliers capped using IQR method")

            # ---------------- Feature Engineering ----------------
            for df in [train_df, test_df]:
                df['Hypertension'] = df.apply(lambda x: self.blood_pressure_classification(x['sysbp'], x['diabp']), axis=1)
                df['Diabetes_grade'] = df['glucose'].apply(self.diabetes_grade)
                df['mean_art_pressure'] = (df['sysbp'] + 2 * df['diabp']) / 3
                df['bp_diff'] = df['sysbp'] - df['diabp']
                df['age_bmi'] = df['age'] * df['bmi']

                # Encode categorical
                df['sex'] = df['sex'].map({'F': 0, 'M': 1}).astype('category')
                df['is_smoking'] = df['is_smoking'].astype('category')
                df['is_smoking_num'] = df['is_smoking'].cat.codes
                df['smoking_intensity'] = df['cigsperday'] * df['is_smoking_num']

                # Cast categories
                df['Hypertension'] = df['Hypertension'].astype('category')
                df['Diabetes_grade'] = df['Diabetes_grade'].astype('category')
                df['education'] = df['education'].astype('int')

            logging.info("Feature engineering done")

            # ---------------- Target Split ----------------
            target_column = 'tenyearchd'
            X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
            X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

            # ---------------- Feature Groups ----------------
            num_features = ['age','cigsperday','totchol','sysbp','diabp','bmi','heartrate','glucose',
                            'bp_diff','age_bmi','smoking_intensity','mean_art_pressure']
            ord_features = ['education']
            cat_features = ['sex', 'is_smoking_num']
            bin_features = ['prevalentstroke','diabetes']
            new_cat_features = ['Hypertension','Diabetes_grade']

            # ---------------- Pipelines ----------------
            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])

            ord_pipeline = Pipeline([
                ('ordinal', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_features),
                ('ord', ord_pipeline, ord_features),
                ('cat', cat_pipeline, cat_features),
                ('bin', 'passthrough', bin_features),
                ('new_cat', 'passthrough', new_cat_features)
            ])

            # Transform data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            logging.info("Preprocessing completed")

            # Save preprocessor
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)

            # ---------------- Handle Imbalance ----------------
            if self.use_smoteenn:
                sampler = SMOTEENN(random_state=42)
                logging.info("Using SMOTEENN for resampling")
            else:
                sampler = SMOTE(random_state=42)
                logging.info("Using SMOTE for resampling")

            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train)

            logging.info("Resampling applied")

            return (
                X_train_resampled, y_train_resampled,
                X_test_processed, y_test,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise TenYearChdException(e, sys)
