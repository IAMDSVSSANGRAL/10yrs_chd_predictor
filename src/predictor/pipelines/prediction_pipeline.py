import os
import sys
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging


@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.model = None
        self.preprocessor = None

    def load_artifacts(self):
        """
        Load model and preprocessor
        """
        try:
            if not os.path.exists(self.config.model_path):
                raise Exception(f"Model not found at {self.config.model_path}")
            
            if not os.path.exists(self.config.preprocessor_path):
                raise Exception(f"Preprocessor not found at {self.config.preprocessor_path}")
            
            self.model = joblib.load(self.config.model_path)
            self.preprocessor = joblib.load(self.config.preprocessor_path)
            
            logging.info("Artifacts loaded successfully")
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def preprocess_input(self, input_data: pd.DataFrame):
        """
        Preprocess input data using the saved preprocessor
        """
        try:
            # Apply the same transformations as in training
            processed_data = self.preprocessor.transform(input_data)
            return processed_data
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def predict(self, input_data: pd.DataFrame):
        """
        Make predictions on input data
        """
        try:
            # Load artifacts if not loaded
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            # Preprocess
            processed_data = self.preprocess_input(input_data)
            
            # Predict
            predictions = self.model.predict(processed_data)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                return predictions, probabilities
            
            return predictions, None
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def predict_single(self, features: dict):
        """
        Make prediction for a single sample
        """
        try:
            # Convert dict to DataFrame
            input_df = pd.DataFrame([features])
            
            # Make prediction
            prediction, probabilities = self.predict(input_df)
            
            result = {
                'prediction': int(prediction[0]),
                'risk_level': 'High Risk' if prediction[0] == 1 else 'Low Risk'
            }
            
            if probabilities is not None:
                result['probability_low_risk'] = float(probabilities[0][0])
                result['probability_high_risk'] = float(probabilities[0][1])
            
            return result
            
        except Exception as e:
            raise TenYearChdException(e, sys)


class CustomData:
    """
    Class for handling custom input data
    """
    def __init__(self,
                 age: int,
                 sex: str,
                 is_smoking: str,
                 cigsperday: float,
                 bpmeds: float,
                 prevalentstroke: int,
                 prevalenthyp: int,
                 diabetes: int,
                 totchol: float,
                 sysbp: float,
                 diabp: float,
                 bmi: float,
                 heartrate: float,
                 glucose: float,
                 education: int):
        
        self.age = age
        self.sex = sex
        self.is_smoking = is_smoking
        self.cigsperday = cigsperday
        self.bpmeds = bpmeds
        self.prevalentstroke = prevalentstroke
        self.prevalenthyp = prevalenthyp
        self.diabetes = diabetes
        self.totchol = totchol
        self.sysbp = sysbp
        self.diabp = diabp
        self.bmi = bmi
        self.heartrate = heartrate
        self.glucose = glucose
        self.education = education

    def get_data_as_dataframe(self):
        """
        Convert custom data to DataFrame
        """
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'sex': [self.sex],
                'is_smoking': [self.is_smoking],
                'cigsperday': [self.cigsperday],
                'bpmeds': [self.bpmeds],
                'prevalentstroke': [self.prevalentstroke],
                'prevalenthyp': [self.prevalenthyp],
                'diabetes': [self.diabetes],
                'totchol': [self.totchol],
                'sysbp': [self.sysbp],
                'diabp': [self.diabp],
                'bmi': [self.bmi],
                'heartrate': [self.heartrate],
                'glucose': [self.glucose],
                'education': [self.education]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise TenYearChdException(e, sys)