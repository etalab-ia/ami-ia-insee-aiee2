"""
Classe mère des Process (transformation de l'input textuel en représentation vectorielle)

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import gensim
import tempfile
import pickle
from datetime import datetime
import pandas as pd
from . import utils
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod

from abc import ABC

class Process(ABC):
    
    def __init__(self):
        pass
    
    ##### Remove because broken in children classes

    # @abstractmethod
    # def save_model():
    #     pass
        
    # @abstractmethod
    # def load_model():
    #     pass
