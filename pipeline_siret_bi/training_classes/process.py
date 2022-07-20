"""
Classe mère des process, qui transforment les input NLP en inputs vectoriels

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import gensim
import tempfile
import pickle
from datetime import datetime
import pandas as pd
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from abc import ABC

from .utils import *


class Process(ABC):
    
    def __init__(self):
        pass
    
    ### Remove because broken in ProcessMLPSiamese
    
    # @abstractmethod
    # def save_model():
    #     pass
        
    @abstractmethod
    def run(self, input_df_path, output_file):
        pass
