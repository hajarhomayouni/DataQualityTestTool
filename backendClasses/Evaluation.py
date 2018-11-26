import csv
import pandas as pd

class Evaluation:
    @staticmethod
    def csvToSet(csvFile):
        recordSet=set()
        with open(csvFile, 'rt') as csvFileRead:
            next(csvFileRead)
            spamreader = csv.reader(csvFileRead, delimiter=',')
            for row in spamreader2:
                recordSet=recordSet.union(set((row[0])))
        return recordSet

    def previouslyDetectedFaultyRecords(A, E):
        common=E.intersection(A)
        return float(len(common))/ float(len(E))

    def newlyDetectedFaultyRecords(A, PD, TF):
        return float(len(TF.difference(PD)))/float(len(A))

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

    def truePositiveRateByTime(datasetInfo):
        return 1

    def invalidityScoreByRecordId(datasetInfo):
        return 1
