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
        train_arr, test_arr,_= data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )

        logging.info("Data Transformation completed successfully")

    except Exception as e:
        logging.info("Custom Exception")
        raise TenYearChdException(e, sys)
