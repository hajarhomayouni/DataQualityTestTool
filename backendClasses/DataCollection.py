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
        dataFrame=dataFrame.fillna("NULL")

        """categoricalColumns=[]
        for column in dataFrame.columns:
            if dataFrame[column].dtype != np.number:
                dataFrame[column]=dataFrame[column].apply(hash)
            if all(float(x).is_integer() for x in dataFrame[column]):
                categoricalColumns.append(column)            


        min_max=MinMaxScaler()
        le=LabelEncoder()
        for col in categoricalColumns:
            data=dataFrame[col]
            le.fit(data.values)
            dataFrame[col]=le.transform(dataFrame[col])
        dataFrame[dataFrame.columns.values]=min_max.fit_transform(dataFrame[dataFrame.columns.values])"""       
        return dataFrame



    
    def findCategorical(self,df_data):
        categorical_columns=[]
        for column in df_data.columns.values:
            if self.is_number(df_data.iloc[1][column])==False:
                print column
                print "%%%%%%%%%%%%%%%%%%%%%%"
                #df_data[column]=df_data[column].apply(hash)
                categorical_columns.append(column)
            #if all(float(x).is_integer() for x in df_data[column]):
            #    categorical_columns.append(column)
        return categorical_columns
    
    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False    

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
        #plt.tick_params(labelsize=1)
        plt.rcParams.update({'font.size': 5})
        plt.tight_layout()
        plt.figure(figsize=(30,3))
        plt.plot(x_coordinates, y_coordinates,'o')
        plt.savefig(img, format='png',bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)
