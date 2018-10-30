import abc
import numpy as np
import pandas as pd


class Testing:

    @abc.abstractmethod
    def assignInvalidityScore(model, testDataFrame):
        pass
    
    @staticmethod
    def detectFaultyRecords(testDataFrame, invalidityScores, threshold):
        faultyRecordsFrame=pd.DataFrame(columns=list(testDataFrame.columns.values))
        outlier_indexes, = np.where(invalidityScores > threshold)
        for outlier in range(len(outlier_indexes)):
            faultyRecordsFrame.loc[outlier]=testDataFrame.iloc[outlier_indexes[outlier]]
        faultyRecordsFrame['invalidityScore']=invalidityScores[outlier_indexes]
        return faultyRecordsFrame

    @abc.abstractmethod
    def clusterFaultyRecords(clusterModel, faultyRecordsFrame):
        pass
    
        

    
