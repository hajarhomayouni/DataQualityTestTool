from DataCollection import DataCollection
from Autoencoder import Autoencoder
from Testing import Testing
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import numpy as np



#work with classes
dataCollection= DataCollection()
dataFrame=dataCollection.selectFeatures(1)
print dataFrame


#Read csv data
dataFrame=dataCollection.importData("testData.csv").head(100)

#Preprocess data - The first column is assumed to be an ID column
dataFramePreprocessed= dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1), ['gender_concept_id'], ['year_of_birth'])


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
print testing.detectFaultyRecords(dataFrame, invalidityScores, np.median(invalidityScores))
