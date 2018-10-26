from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

class DataCollection:

    def importData(dataPath):
        dataFrame="" #read CSV to dataframe
        return dataFrame
    
    def preprocess(dataFrame,categoricalColumns,numericColumns):
        #proprocess null data
        dataFrame=dataFrame.fillna(99999)
        #preprocess numeric columns
        min_max=MinMaxScaler()
        dataFrame[numericColumns]=min_max.fit_transform(dataFrame[numericColumns])
        #preprocess categorical columns
        le=LabelEncoder()
        for col in categoricalColumns:
            data=dataFrame[col]
            le.fit(data.values)
            dataFrame[col]=le.transform(dataFrame[col])
        return dataFrame


    def selectFeatures(dataFrame):
        #use feature selection/prioritization methods
        return dataFrame
