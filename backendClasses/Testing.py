import abc
import numpy as np
import pandas as pd
import statistics


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

    @staticmethod
    def detectNormalRecordsBasedOnFeatures(testDataFrame, invalidityScoresPerFeature, invalidityScores, threshold):
        normalFrame=pd.DataFrame(columns=list(testDataFrame.columns.values))
        normal,=np.where(invalidityScoresPerFeature[1]<=statistics.median(invalidityScoresPerFeature[1]))
        normal_indexes=set(normal)
        for col in invalidityScoresPerFeature.columns.values[2:]:
            normal, =np.where(invalidityScoresPerFeature[col]<=statistics.median(invalidityScoresPerFeature[col])) #for col in testDataFrame.columns.values))
            normal_indexes.intersection(normal)
        #excule the indexes of the records that are reported as faulty
        normal_indexes2, = np.where(invalidityScores <= threshold)
        normal_indexes=normal_indexes.intersection(set(normal_indexes2))
        
        for normal in range(len(list(normal_indexes))):
            normalFrame.loc[normal]=testDataFrame.iloc[list(normal_indexes)[normal]]

        normalFrame['invalidityScore']=invalidityScores[list(normal_indexes)]
        return normalFrame


    
    @abc.abstractmethod
    def clusterFaultyRecords(self, faultyRecordFramePreprocessed, faultyRecordFrame):
        pass

    
        

    
