from PatternDiscovery import PatternDiscovery
import h2o
from h2o.grid.grid_search import H2OGridSearch
from sklearn import preprocessing
import pandas as pd

class Autoencoder(PatternDiscovery):


    @staticmethod
    def tuneAndTrain(hyperParameters, model, trainDataFrame):
        h2o.init()
        trainData=trainDataFrame.values        
        trainDataHex=h2o.H2OFrame(trainData)
        modelGrid = H2OGridSearch(model,hyper_params=hyperParameters)
        modelGrid.train(x= list(range(0,int(len(trainDataFrame.columns)))),training_frame=trainDataHex)
        gridperf1 = modelGrid.get_grid(sort_by='mse', decreasing=True)
        bestModel = gridperf1.models[0]
        return bestModel

    @staticmethod
    def assignInvalidityScore(model, testDataFrame):
        testData=testDataFrame.values        
        testDataHex=h2o.H2OFrame(testData)
        recon_error = model.anomaly(testDataHex)
        recon_error_np=recon_error.as_data_frame().values
        recon_error_preprocessed=preprocessing.normalize(recon_error_np, norm='l2',axis=0)
        return recon_error_preprocessed.ravel()

    @staticmethod
    def assignInvalidityScorePerFeature(model, testDataFrame):
        testData=testDataFrame.values        
        testDataHex=h2o.H2OFrame(testData)
        recon_error = model.anomaly(testDataHex, per_feature = True)
        recon_error_preprocessed=preprocessing.normalize(recon_error.as_data_frame(), norm='l2',axis=0)
        return pd.DataFrame.from_records(recon_error_preprocessed)

