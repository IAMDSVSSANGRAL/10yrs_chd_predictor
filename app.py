from src.predictor.logger import logging
from src.predictor.exception import TenYearChdException
from src.predictor.components.data_ingestion import DataIngestion
from src.predictor.components.data_ingestion import DataIngestionConfig
import os
import sys

if __name__ =="__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()


    except Exception as e:
        logging.info("Custom Exception")
        raise TenYearChdException(e,sys)