from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

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
    
    def preprocess2(df_data):
        categorical_columns=[]
        for column in df_data.columns:
            if df_data[column].dtype != np.number:
                df_data[column]=df_data[column].apply(hash)
            if all(float(x).is_integer() for x in df_data[column]):
                categorical_columns.append(column)   

        #preprocess categorical columns
        df_data_1=df_data.drop(categorical_columns, axis=1)   
        enc=OneHotEncoder(sparse=False) 
        for col in categorical_columns:
            # creating an exhaustive list of all possible categorical values
            data=df_data[[col]]
            enc.fit(data)
            # Fitting One Hot Encoding on train data
            temp = enc.transform(df_data[[col]])
            # Changing the encoded features into a data frame with new column names
            temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
                                            .value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the X_train data frame
            temp=temp.set_index(df_data.index.values)
            # adding the new One Hot Encoded varibales to the train data frame
            df_data_1=pd.concat([df_data_1,temp],axis=1)
        return df_data_1
    
    
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

    @staticmethod
    def build_graph(x_coordinates, y_coordinates):
        img = io.BytesIO()
        #plt.xticks(rotation=45)
        plt.plot(x_coordinates, y_coordinates,'o')
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)
