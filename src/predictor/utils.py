import os
import sys
import pandas as pd
import psycopg2
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
        df = pd.read_sql_query("""select * from data_cardiovascular_risk;""", connection)
        logging.info("Data reading completed successfully")
        print(df.shape)
        
        # Close the connection
        connection.close()
        return df
    
    except Exception as ex:
        logging.error("Error while reading data from PostgreSQL", exc_info=True)
        raise TenYearChdException(ex, sys)
