from DataCollection import DataCollection
from Autoencoder import Autoencoder
from Testing import Testing
from SOM import SOM
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import numpy as np
import pandas as pd



#work with classes
dataCollection= DataCollection()
dataFrame=dataCollection.selectFeatures(1)
print dataFrame


#Read csv data
dataFrame=dataCollection.importData("testData.csv").head(100)

#Preprocess data 
dataFramePreprocessed= dataCollection.preprocess(dataFrame, ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])       


#Tune and Train model - The first column is assumed to be an ID column
autoencoder = Autoencoder()
hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
l2Opt = [1e-4,1e-2]
hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=100),dataFramePreprocessed.drop([dataFramePreprocessed.columns.values[0]], axis=1))

#Assign invalidity scores - The first column is assumed to be an ID column
invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed.drop([dataFramePreprocessed.columns.values[0]], axis=1))

#Detect faulty records
testing=Testing()
faultyRecordsFrame=testing.detectFaultyRecords(dataFramePreprocessed, invalidityScores, np.median(invalidityScores))


#Cluster the faulty records
#Exclude id columnand invalidity score for clustering
som = SOM(5,5, len(faultyRecordsFrame.columns.values)-2, 400)
print som.clusterFaultyRecords(faultyRecordsFrame.columns.values[0],'invalidityScore'],axis=1))



