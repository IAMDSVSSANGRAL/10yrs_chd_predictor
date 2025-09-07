from src.predictor.logger import logging
from src.predictor.exception import TenYearChdException
from src.predictor.components.data_ingestion import DataIngestion
from src.predictor.components.data_transformation import DataTransformation, DataTransformationConfig
import os
import sys

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )

        logging.info("Data Transformation completed successfully")
        logging.info(f"Preprocessor saved at: {preprocessor_path}")

        # Optionally, you can add further steps here to save your resampled data or proceed to model training
        # For example:
        # save_object('artifacts/X_train_resampled.pkl', X_train_resampled)
        # save_object('artifacts/y_train_resampled.pkl', y_train_resampled)
        # save_object('artifacts/X_test_processed.pkl', X_test_processed)
        # save_object('artifacts/y_test.pkl', y_test)

    except Exception as e:
        logging.error("Error in execution")
        raise TenYearChdException(e, sys)
