import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging

@dataclass
class DataValidationConfig:
    report_file_path: str = os.path.join("artifacts", "data_validation_report.txt")

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_schema(self, df: pd.DataFrame):
        """
        Validate dataset schema: check column presence and dtypes
        """
        expected_columns = {
            'id': 'int64',
            'age': 'int64',
            'sex': 'object',
            'is_smoking': 'object',
            'cigsperday': 'float64',
            'bpmeds': 'float64',
            'prevalentstroke': 'int64',
            'prevalenthyp': 'int64',
            'diabetes': 'int64',
            'totchol': 'float64',
            'sysbp': 'float64',
            'diabp': 'float64',
            'bmi': 'float64',
            'heartrate': 'float64',
            'glucose': 'float64',
            'education': 'float64',
            'tenyearchd': 'int64'
        }

        errors = []
        for col, dtype in expected_columns.items():
            if col not in df.columns:
                errors.append(f"[X] Missing column: {col}")
            else:
                if str(df[col].dtype) != dtype:
                    errors.append(f"[!] Column {col} expected {dtype} but got {df[col].dtype}")

        return errors

    def validate_values(self, df: pd.DataFrame):
        """
        Validate ranges and categorical domains
        """
        errors = []
        # Age
        if df['age'].min() < 20 or df['age'].max() > 100:
            errors.append("[!] Age values out of expected range (20-100).")

        # Sex
        if not set(df['sex'].dropna().unique()).issubset({'M', 'F'}):
            errors.append("[!] Invalid values in sex column.")

        # Smoking
        if not set(df['is_smoking'].dropna().unique()).issubset({'YES', 'NO'}):
            errors.append("[!] Invalid values in is_smoking column.")

        # BMI sanity check
        if df['bmi'].dropna().min() < 10 or df['bmi'].dropna().max() > 70:
            errors.append("[!] BMI values out of range (10-70).")

        return errors

    def initiate_data_validation(self, raw_data_path: str):
        """
        Main entrypoint to validate data from CSV file path
        """
        try:
            logging.info("Starting data validation")
            
            # Read the raw data
            df = pd.read_csv(raw_data_path)
            logging.info(f"Read data from {raw_data_path} with shape {df.shape}")

            schema_errors = self.validate_schema(df)
            value_errors = self.validate_values(df)

            all_errors = schema_errors + value_errors
            if not all_errors:
                validation_status = "[OK] Data Validation PASSED"
            else:
                validation_status = "[FAILED] Data Validation FAILED"

            # Save report with UTF-8 encoding
            os.makedirs(os.path.dirname(self.validation_config.report_file_path), exist_ok=True)
            with open(self.validation_config.report_file_path, "w", encoding='utf-8') as f:
                f.write(f"Data Validation Report\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Dataset Shape: {df.shape}\n\n")
                if all_errors:
                    f.write("Issues Found:\n")
                    f.write("\n".join(all_errors))
                else:
                    f.write("No issues found.\n")
                f.write(f"\n\n{validation_status}")

            logging.info(validation_status)
            
            # Return boolean indicating if validation passed
            validation_passed = len(all_errors) == 0
            return validation_passed, self.validation_config.report_file_path

        except Exception as e:
            raise TenYearChdException(e, sys)