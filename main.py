from predictor.exception import tenyearchdexception
import sys

def test_exception():
    try:
        a=1/0
    except Exception as e:
        #raise tenyearchdexception(e,sys)
        raise e


if __name__ == "__main__":
    try:
        test_exception()
    except Exception as e:
        print(e)