import psycopg2
from config import Config
import logging
from exception import tenyearchdexception
import sys

# Initialize logger
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using the Config class.
    Logs success or failure accordingly.
    """
    config = Config()
    conn_str = config.get_database_url()

    try:
        connection = psycopg2.connect(conn_str)
        logger.info("Successfully connected to the database.")
        return connection

    except Exception as error:
        logger.error("Error connecting to the database", exc_info=True)
        raise tenyearchdexception(error, sys)


if __name__ == "__main__":
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()
            logger.info(f"Database version: {db_version}")
            print("Database version:", db_version)

    except tenyearchdexception as e:
        logger.critical(f"Critical error occurred: {str(e)}")

    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
