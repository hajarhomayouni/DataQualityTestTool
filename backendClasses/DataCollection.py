from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv

class DataCollection:

    
    @staticmethod
    def importData(csvPath):
        return pd.DataFrame.from_csv(csvPath)

    @staticmethod
    def preprocess(dataFrame):
        #proprocess null data
        dataFrame=dataFrame.fillna(9999999999)

        categoricalColumns=[]
        for column in dataFrame.columns:
            if dataFrame[column].dtype != np.number:
                dataFrame[column]=dataFrame[column].apply(hash)
            if all(float(x).is_integer() for x in dataFrame[column]):
                categoricalColumns.append(column)            


        """min_max=MinMaxScaler()
        le=LabelEncoder()
        for col in categoricalColumns:
            data=dataFrame[col]
            le.fit(data.values)
            dataFrame[col]=le.transform(dataFrame[col])
        dataFrame[dataFrame.columns.values]=min_max.fit_transform(dataFrame[dataFrame.columns.values])"""       
        return dataFrame
    @staticmethod
    def selectFeatures(dataFrame):
        #use feature selection/prioritization methods
        return dataFrame



    @staticmethod
    def csvToSet(csvFile):        
        recordSet=set()
        with open(csvFile, 'rt') as csvFileRead:
            spamreader = csv.reader(csvFileRead, delimiter=',')
            for row in spamreader:
                recordSet=recordSet.union(set(row))
        return recordSet
