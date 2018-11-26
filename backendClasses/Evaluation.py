import csv
import pandas as pd

class Evaluation:
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
    def trulyDetectedFaultyRecords(datasetId, con):
        TFdataFrame=pd.read_sql(sql="select distinct fault_id from TF where dataset_id like '"+datasetId+"'", con=con )
        TFlist=TFdataFrame['fault_id'].unique().tolist()
        return set(TFlist)

    def truePositiveRate(A, TF):
        return float(len(TF))/float(len(A))

    def save(datasetInfo, PD, ND, UD, TP):
        return 1

    def truePositiveRateByTime(datasetInfo):
        return 1

    def invalidityScoreByRecordId(datasetInfo):
        return 1
