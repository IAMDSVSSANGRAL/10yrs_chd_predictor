## postgresql -> Train test split -> dataset
import os
import sys
from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging
import pandas as pd


from dataclasses import dataclass

@dataclass
class Dataingestionconfig:
    tain 