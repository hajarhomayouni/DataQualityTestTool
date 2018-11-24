import csv

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

    def truelyDetectedFaultyRecords(databaseInfo):
        return 1

    def truePositiveRate(A, TF):
        return float(len(TF))/float(len(A))

    def save(datasetInfo, PD, ND, UD, TP):
        return 1

    def truePositiveRateByTime(datasetInfo):
        return 1
