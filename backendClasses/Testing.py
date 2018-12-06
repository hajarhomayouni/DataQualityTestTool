import abc
import numpy as np
import pandas as pd


class Testing:

    @abc.abstractmethod
    def assignInvalidityScore(model, testDataFrame):
        pass
    
    @staticmethod
    def detectFaultyRecords(testDataFrame, invalidityScores, threshold):
        outlierFrame=pd.DataFrame(columns=list(testDataFrame.columns.values))
        outlier_indexes, = np.where(invalidityScores > threshold)
        for outlier in range(len(outlier_indexes)):
            outlierFrame.loc[outlier]=testDataFrame.iloc[outlier_indexes[outlier]]
        outlierFrame['invalidityScore']=invalidityScores[outlier_indexes]
        return outlierFrame
    
    @staticmethod
    def detectNormalRecords(testDataFrame, invalidityScores, threshold):
        normalFrame=pd.DataFrame(columns=list(testDataFrame.columns.values))
        normal_indexes, = np.where(invalidityScores <= threshold)
        for normal in range(len(normal_indexes)):
            normalFrame.loc[normal]=testDataFrame.iloc[normal_indexes[normal]]
        normalFrame['invalidityScore']=invalidityScores[normal_indexes]
        return normalFrame

    @abc.abstractmethod
    def clusterFaultyRecords(self, faultyRecordFramePreprocessed, faultyRecordFrame):
        pass

    
        

    
