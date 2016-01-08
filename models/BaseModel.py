import logging
logger = logging.getLogger("DeepSequenceClassification_Model")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("Started Logger")

import theano, keras
logger.info("Using Keras version %s" % keras.__version__)
logger.info("Using Theano version %s" % theano.__version__)
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Merge, Permute, Reshape, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import accuracy

__author__ = "Shubhanshu Mishra"

class BaseModel:
    def __init__(self):
        self.__model_code__ = "BaseModel"
        raise Exception("Feature not implemented")

    def get_model(self):
        """
        This function should be overriden by every model class with details of the model
        """
        raise Exception("Feature not implemented")

    def preprocess_vectors(self):
        """
        Add code for preprocessing the vectors from raw file
        """
        raise Exception("Feature not implemented")
    
    def save_vectors(self):
        """
        Add code for saving the vectors as numpy array file
        """
        raise Exception("Feature not implemented")
    
    def load_vectors(self):
        """
        If vectors exist load from file else generate vectors and save and load from files
        """
        raise Exception("Feature not implemented")

    def train(self):
        """
        Code for training the model
        """
        raise Exception("Feature not implemented")

    def evaluate(self):
        """
        Evaluate the model
        """
        raise Exception("Feature not implemented")

    def save_model(self):
        """
        Save model
        """
        raise Exception("Feature not implemented")

    def load_model(self):
        """
        Load model
        """
        raise Exception("Feature not implemented")
