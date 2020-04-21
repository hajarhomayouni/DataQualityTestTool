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
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import Binarizer


class DataCollection:

    
    @staticmethod
    def importData(csvPath):
        return pd.read_csv(csvPath,index_col=0)
        #return pd.DataFrame.from_csv(csvPath)

    def preprocess(self,dataFrame):
        #proprocess null data
        dataFrame=dataFrame.fillna(-1)

        """categoricalColumns=[]
        for column in dataFrame.columns:
            if dataFrame[column].dtype != np.number:
                dataFrame[column]=dataFrame[column].apply(hash)
            if all(float(x).is_integer() for x in dataFrame[column]):
                categoricalColumns.append(column)            


        le=LabelEncoder()
        for col in categoricalColumns:
            data=dataFrame[col]
            le.fit(data.values)
            dataFrame[col]=le.transform(dataFrame[col])"""

        """for column in dataFrame.columns:
            #if dataFrame[column].dtype==np.number:
            if self.is_number(dataFrame.iloc[1][column]) and column!="id" and column!="time":
                #1
                min_max=MinMaxScaler(feature_range=(0, 1))
                dataFrame[[column]]=min_max.fit_transform(dataFrame[[column]])
                #2
                #dataFrame[[column]]=preprocessing.normalize(dataFrame[[column]], norm='l1',axis=1)
                #3-best
                #disc = KBinsDiscretizer(n_bins=10, encode='ordinal',strategy='kmeans')
                #dataFrame[[column]]=disc.fit_transform(dataFrame[[column]])
                #4
                #dataFrame[[column]]=scale(dataFrame[[column]])
                #5
                #binarizer=Binarizer(threshold=0.0)
                #dataFrame[[column]]=binarizer.fit_transform(dataFrame[[column]])"""

        print (dataFrame)
        return dataFrame



    
    def findCategorical(self,df_data):
        categorical_columns=[]
        for column in df_data.columns.values:
            if self.is_number(df_data.iloc[1][column])==False:
                i#df_data[column]=df_data[column].apply(hash)
                categorical_columns.append(column)
            elif all(float(x).is_integer() for x in df_data[column]):
                categorical_columns.append(column)
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
        plt.rcParams.update({'font.size': 15})
        plt.tight_layout()
        plt.figure(figsize=(30,3))
        plt.plot(x_coordinates, y_coordinates,'o')
        plt.savefig(img, format='png',bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)
