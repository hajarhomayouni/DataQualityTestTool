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

#Preprocess data - The first column is assumed to be an ID column
dataFramePreprocessed= dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1), ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])       


#Tune and Train model
autoencoder = Autoencoder()
hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
l2Opt = [1e-4,1e-2]
hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=100),dataFramePreprocessed)

#Assign invalidity scores
invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)

#Detect faulty records
testing=Testing()
outlierFrame=testing.detectFaultyRecords(dataFrame, invalidityScores, np.median(invalidityScores))
outlierFramePreprocessed=dataCollection.preprocess(outlierFrame.drop([outlierFrame.columns.values[0],'invalidityScore'],axis=1), ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high']) 


#Cluster the faulty records
#Train a 5*5 SOM with 100 iterations
#Exclude id columnand invalidity score for clustering
som = SOM(5,5, len(outlierFrame.columns.values)-2, 400)
som.train(outlierFramePreprocessed.values) 
#Map data to their closest neurons
mapped= som.map_vects(outlierFramePreprocessed.values)
labels_list=[list(x) for x in mapped]
print labels_list

groups_of_outliers=[]
group_index=0
for i in range(5):
    for j in range(5):
        indexes_in_cluster=[k for k, x in enumerate(labels_list) if x == [i,j]]
        print indexes_in_cluster
        if len(outlierFrame.values[indexes_in_cluster]>0):
            groups_of_outliers.append(outlierFrame.iloc[indexes_in_cluster])
            group_index+=1


print groups_of_outliers
print "*********************"
print groups_of_outliers[0]
