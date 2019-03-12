from h2o.estimators.kmeans import H2OKMeansEstimator
from Testing import Testing
import h2o
from h2o.grid.grid_search import H2OGridSearch
from sklearn import preprocessing
import pandas as pd
from DataCollection import DataCollection
import numpy as np

class H2oKmeans(Testing):


    @staticmethod
    def tuneAndTrain(trainDataFrame):
        h2o.init()
        #trainData=trainDataFrame       
        trainDataHex=h2o.H2OFrame(trainDataFrame)
        #to consider categorical columns uncomment all the comments
        """dc=DataCollection()
        categoricalColumns=dc.findCategorical(trainDataFrame)
        trainDataHex[categoricalColumns] = trainDataHex[categoricalColumns].asfactor()"""
        #
        k = range(2,20)
        hyperParameters = {"k":k}
        modelGrid = H2OGridSearch(H2OKMeansEstimator(ignore_const_cols=False),hyper_params=hyperParameters)
        modelGrid.train(x= list(range(0,int(len(trainDataFrame.columns)))),training_frame=trainDataHex)
        gridperf1 = modelGrid.get_grid(sort_by='mse', decreasing=True)
        bestModel = gridperf1.models[0]
        return bestModel
        #
        """model = H2OKMeansEstimator(k = 5, estimate_k = True, ignore_const_cols=False)
        model.train(x= list(range(0,int(len(trainDataFrame.columns)))),training_frame=trainDataHex)
        return model"""

    @staticmethod
    def clusterFaultyRecords(model,testDataFramePreprocessed, testDataFrame):
        testDataHex=h2o.H2OFrame(testDataFramePreprocessed)
        #to consider categorical columns uncomment all the comments
        """dc=DataCollection()
        categoricalColumns=dc.findCategorical(testDataFramePreprocessed)
        testDataHex[categoricalColumns] = testDataHex[categoricalColumns].asfactor()"""
        #
        labels_test=h2o.as_list(model.predict(testDataHex))["predict"].tolist()
        print type(labels_test)
        groups_of_faultyRecords=[]
        centroids=len(model.centers())
        for cluster in range(centroids):
            cluster_indexes, = np.where(np.asarray(labels_test) == cluster)
            groups_of_faultyRecords.append(testDataFrame.iloc[cluster_indexes])
        return groups_of_faultyRecords



        


