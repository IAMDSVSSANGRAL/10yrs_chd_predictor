## postgresql -> Train test split -> dataset
import os
import sys
from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging
from src.predictor.utils import read_sql_data
from sklearn.model_selection import train_test_split
import pandas as pd


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str  = os.path.join('artifiacts','train.csv')
    test_data_path:str  = os.path.join('artifiacts','test.csv')
    raw_data_path:str  = os.path.join('artifiacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            #reading the code from postgresql
            query="""select * from data_cardiovascular_risk;"""
            df=read_sql_data(query)

            logging.info("Reading from postgresql database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=66)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)

        except Exception as e:
            raise TenYearChdException(e,sys)