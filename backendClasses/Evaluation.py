import csv
import pandas as pd

class Evaluation:

    @staticmethod
    def previouslyDetectedFaultyRecords(A, E):
        common=E.intersection(A)
        if len(E)>0:
            return float(len(common))/ float(len(E))
        else:
            return 0.0

    @staticmethod
    def newlyDetectedFaultyRecords(A, E, TF):
        return float(len(TF.difference(E)))/float(len(A))

    @staticmethod
    def unDetectedFaultyRecords(A, E):
        if len(E)>0:
            return float(len(E.difference(A)))/float(len(E))
        else:
            return 1.0

    @staticmethod
    def truePositiveGrowthRate(score):
        beginingTPR=score['TPR'].iloc[0]
        endingTPR=score['TPR'].iloc[-1]
        NR=float(len(score['TPR'].tolist()))
        return ((endingTPR/beginingTPR)**(1/NR))-1
    
    @staticmethod
    def numberOfRuns(score):
        return float(len(score['TPR'].tolist()))

    @staticmethod
    def truePositiveRate(A, TF):
        return float(len(TF))/float(len(A))

    @staticmethod
    def truePositiveRateByTime(datasetInfo):
        return 1

    @staticmethod
    def invalidityScoreByRecordId(datasetInfo):
        return 1
