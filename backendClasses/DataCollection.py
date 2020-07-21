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
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class DataCollection:

    
    @staticmethod
    def importData(csvPath):
        return pd.read_csv(csvPath,index_col=0, error_bad_lines=False)
        #return pd.DataFrame.from_csv(csvPath)

    def preprocess(self,dataFrame,grouping_attr=None):
        #proprocess null data
        dataFrame=dataFrame.fillna(-1)


        categorical_feature_mask, categoricalColumns = self.find_categorical(dataFrame)
        if grouping_attr and grouping_attr in categoricalColumns:
            categoricalColumns.remove(grouping_attr)
        
        #1. similar to one-hot encoding
        if len(categoricalColumns)>0:
            tempdf=pd.get_dummies(dataFrame[categoricalColumns],columns=categoricalColumns)
            dataFrame=dataFrame.drop(categoricalColumns,axis=1)
            dataFrame=pd.concat([dataFrame, tempdf], axis=1)

        #2. remove categorical columns
        #dataFrame=dataFrame.drop([categoricalColumns], axis=1)
        """for col in categoricalColumns:
            if col!="time":
                del dataFrame[col]"""        

        #3. labelencoding
        """le = LabelEncoder()
        for col in categoricalColumns:
            if col!="id" and col!="time":
                dataFrame[col] = dataFrame[col].astype('str')
                le.fit(dataFrame[col])
                dataFrame[col]=le.transform(dataFrame[col])"""
                
        #4. One-hot encoding
        """ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False )
        dataFrame=ohe.fit_transform(dataFrame)"""


        for column in dataFrame.columns:
            #if dataFrame[column].dtype==np.number:
            #if self.is_number(dataFrame.iloc[1][column]) and column!="id" and column!="time":
            if  is_numeric_dtype(dataFrame[column]) and column!="id" and column!="time":
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
                #dataFrame[[column]]=binarizer.fit_transform(dataFrame[[column]])

        print (dataFrame)
        return dataFrame


    def findSubarray(array, subarray):
        len_b = len(subarray)
        for i in range(len(array)):  
            if array[i:i+len_b] == subarray:
                return i,i+len_b 


    def find_categorical(self, dataFrame):
        dataFrame=dataFrame.fillna(-1)
        categorical_feature_mask = dataFrame.dtypes==object
        categoricalColumns = dataFrame.columns[categorical_feature_mask].tolist()
        if "time" in categoricalColumns:
            categoricalColumns.remove("time")
        return categorical_feature_mask,categoricalColumns

    
    
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
    def build_graph(x_coordinates, y_coordinates,font_size=15,x_rotate=0,y_title=None,x_red=None,y_red=None):
        img = io.BytesIO()
        #plt.tick_params(labelsize=1)
        plt.rcParams.update({'font.size': font_size})
        plt.tight_layout()
        plt.figure(figsize=(30,3))
        plt.plot(x_coordinates, y_coordinates,"o")
        plt.xticks(rotation=x_rotate)
        plt.title(y_title)
        #
        if  x_red is not None:
            plt.plot(x_red, y_red,"o", color="red")
        #
        plt.savefig(img, format='png',bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)
