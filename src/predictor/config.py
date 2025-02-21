from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
import logging
from exception import TenYearChdException
import sys

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class Config:
    try:
        HOST: str = field(default_factory=lambda: os.getenv('HOST', 'localhost'))
        PORT: str = field(default_factory=lambda: os.getenv("PORT", "5432"))
        DB: str = field(default_factory=lambda: os.getenv("DATABASE", "your_database_name"))
        USER: str = field(default_factory=lambda: os.getenv("USER", "postgres"))
        PASSWORD: str = field(default_factory=lambda: os.getenv("PASSWORD", "your_password"))
        
        logger.info("Config values loaded successfully.")

    except Exception as e:
        logger.error("Error loading config values", exc_info=True)
        raise TenYearChdException(e, sys)

    def get_database_url(self) -> str:
        """
        Constructs and returns the PostgreSQL connection URL.
        """
        try:
            db_url = (f"postgresql://{self.USER}:{self.PASSWORD}"
                      f"@{self.HOST}:{self.PORT}/{self.DB}")
            logger.info("Database URL constructed successfully.")
            return db_url
        except Exception as e:
            logger.error("Error constructing database URL", exc_info=True)
            raise TenYearChdException(e, sys)
