import os
import sys
import pandas as pd
import psycopg2
import pickle
from dotenv import load_dotenv
from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging

# Load environment variables
load_dotenv()

# Get database credentials from environment variables
host = os.getenv("HOST")
user = os.getenv("USER")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
password = os.getenv("PASSWORD")

def read_sql_data(query: str):
    """Reads data from a PostgreSQL database and returns a pandas DataFrame."""
    logging.info("Reading PostgreSQL database started")
    
    try:
        # Establish the database connection
        connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        logging.info(f"Connection Established name {connection}")
        # Read data into a DataFrame
        df = pd.read_sql_query(query, connection)
        logging.info("Data reading completed successfully")
        print(df.shape)

        # Close the connection
        connection.close()
        return df
    
    except Exception as ex:
        logging.error("Error while reading data from PostgreSQL", exc_info=True)
        raise TenYearChdException(ex, sys)

def save_object(file_path, obj):
    """Saves a Python object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")

    except Exception as ex:
        logging.error("Error while saving object", exc_info=True)
        raise TenYearChdException(ex, sys)

def read_csv_data(file_path):
    """Reads CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV file {file_path} read successfully")
        return df
    except Exception as ex:
        logging.error("Error while reading CSV file", exc_info=True)
        raise TenYearChdException(ex, sys)
