import csv
import pandas as pd

class Evaluation:

    @staticmethod
    def previouslyDetectedFaultyRecords(A, E):
        common=E.intersection(A)
        return float(len(common))/ float(len(E))

    @staticmethod
    def newlyDetectedFaultyRecords(A, E, TF):
        return float(len(TF.difference(E)))/float(len(A))

    @staticmethod
    def unDetectedFaultyRecords(A, E):
        return float(len(E.difference(A)))/float(len(E))

    @staticmethod
    def truePositiveGrowthRate(score):
        beginingTPR=score['true_positive_rate'].iloc[0]
        endingTPR=score['true_positive_rate'].iloc[-1]
        NR=float(len(score['true_positive_rate'].tolist()))
        return ((endingTPR/beginingTPR)**(1/NR))-1
    
    @staticmethod
    def numberOfRuns(score):
        return float(len(score['true_positive_rate'].tolist()))

    @staticmethod
    def truePositiveRate(A, TF):
        return float(len(TF))/float(len(A))

    @staticmethod
    def truePositiveRateByTime(datasetInfo):
        return 1

    @staticmethod
    def invalidityScoreByRecordId(datasetInfo):
        return 1
