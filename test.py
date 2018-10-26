from DataCollection import DataCollection
from Autoencoder import Autoencoder
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator



#work with classes
dataCollection= DataCollection()
dataFrame=dataCollection.selectFeatures(1)
print dataFrame


#Read csv data
dataFrame=dataCollection.importData("testData.csv")
print dataCollection.preprocess(dataFrame, ['measurement_concept_id','gender_concept_id'], ['year_of_birth'])


#test interface classes
autoencoder = Autoencoder()
hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
l2Opt = [1e-4,1e-2]
h2o.init()
hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
print autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=100),dataFrame)
