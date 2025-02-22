import logging
import os
import sys
from datetime import datetime

#Define log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR,exist_ok=True)

#Generate log file name dynamically
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE)

#Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)

#ADD CONSOLE LOGGING 
console_handler =logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

#Example Log message
logger = logging.getLogger("TenYearCHDLOGGER")
logger.info("Logging setup complete!")