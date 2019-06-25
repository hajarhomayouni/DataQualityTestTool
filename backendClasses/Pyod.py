from PatternDiscovery import PatternDiscovery
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

class Pyod(PatternDiscovery):


    @staticmethod
    def tuneAndTrain(hyperParameters, model, trainDataFrame):
        #detector_list = [LOF(), LOF()]
        #model=LSCP(detector_list)
        #model=SO_GAAL()
        """if model==AutoEncoder():
            n=len(trainDataFrame.columns.values)
            hidden_neurons=[n-1,n-2,n-2,n-1]
            model=AutoEncoder(hidden_neurons)"""
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

