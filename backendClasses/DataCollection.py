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
        dataFrame=dataFrame.fillna(9999999999)

        categoricalColumns=[]
        numericColumns=[]
        for column in dataFrame.columns:
            if all(float(x).is_integer() for x in dataFrame[column]):
                categoricalColumns.append(column)            
            else:
                numericColumns.append(column)


        print categoricalColumns
        print numericColumns
        #print dataFrame
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
        dataFrame[categoricalColumns]=min_max.fit_transform(dataFrame[categoricalColumns])        
        return dataFrame
    @staticmethod
    def selectFeatures(dataFrame):
        #use feature selection/prioritization methods
        return dataFrame
