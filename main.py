from src.predictor.exception import tenyearchdexception
from src.predictor.logger import logging
import sys

def test_exception():
    try:
        logging.info("We will see one error here.")
        a=1/0
    except Exception as e:
        raise tenyearchdexception(e,sys)
        


if __name__ == "__main__":
    try:
        test_exception()
    except Exception as e:
        print(e)