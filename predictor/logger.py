import logging
import os
import sys
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

os.path.join(os.getcwd(),"logs",LOG_FILE)

logging.basicConfig(filename=LOG_FILE,
                    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s -%(message)s",
                    level=logging.INFO()

)