from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class DataCollection:

    
    @staticmethod
    def importData(csvPath):
        return pd.DataFrame.from_csv(csvPath)

    @staticmethod
    def preprocess(dataFrame):
        #proprocess null data
        dataFrame=dataFrame.fillna(99999)

        categoricalColumns=[]
        numericColumns=[]
        for column in dataFrame.columns:
            if dataFrame[column].dtype == np.float64:
                categoricalColumns.append(column)
            elif dataFrame[column].dtype == np.int64:
                numericColumns.append(column)


        #preprocess numeric columns
        min_max=MinMaxScaler()
        if numericColumns:
            dataFrame[numericColumns]=min_max.fit_transform(dataFrame[numericColumns])
        #preprocess categorical columns
        le=LabelEncoder()
        for col in categoricalColumns:
            data=dataFrame[col]
            le.fit(data.values)
            dataFrame[col]=le.transform(dataFrame[col])
        return dataFrame
    @staticmethod
    def selectFeatures(dataFrame):
        #use feature selection/prioritization methods
        return dataFrame
