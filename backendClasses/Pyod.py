from .PatternDiscovery import PatternDiscovery
import pyod
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
import pandas as pd
from keras.models import load_model
from keras import backend as K

class Pyod(PatternDiscovery):


    @staticmethod
    def tuneAndTrain(model, trainDataFrame):
        K.clear_session()
        model.fit(trainDataFrame)
        return model

    @staticmethod
    def assignInvalidityScore(model, testDataFrame):
        if  len(list(model.decision_function(testDataFrame).shape))>1:
            return model.decision_function(testDataFrame).ravel()
        else:
            return model.decision_function(testDataFrame)


    @staticmethod
    def assignInvalidityScorePerFeature(model, testDataFrame):
        return pd.DataFrame(model.decision_function(testDataFrame), columns=['invalidityScore'])

