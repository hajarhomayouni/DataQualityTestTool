from .PatternDiscovery import PatternDiscovery
import h2o
from h2o.grid.grid_search import H2OGridSearch
from sklearn import preprocessing
import pandas as pd
from .DataCollection import DataCollection

class Autoencoder(PatternDiscovery):


    @staticmethod
    def tuneAndTrain(model, trainDataFrame):
        h2o.init()
        trainDataHex=h2o.H2OFrame(trainDataFrame)
        model.train(x= list(range(0,int(len(trainDataFrame.columns)))),training_frame=trainDataHex)
        print("MSE*****%%%")
        print(model.mse())
        return model

    @staticmethod
    def assignInvalidityScore(model, testDataFrame):
        #testData=testDataFrame.values        
        testDataHex=h2o.H2OFrame(testDataFrame)
        #
        """dc=DataCollection()
        categoricalColumns=dc.findCategorical(testDataFrame)
        testDataHex[categoricalColumns] = testDataHex[categoricalColumns].asfactor()"""
        #
        recon_error = model.anomaly(testDataHex)
        recon_error_np=recon_error.as_data_frame().values
        recon_error_preprocessed=preprocessing.normalize(recon_error_np, norm='l2',axis=0)
        return recon_error_preprocessed.ravel()

    @staticmethod
    def assignInvalidityScorePerFeature(model, testDataFrame):
        #testData=testDataFrame.values        
        testDataHex=h2o.H2OFrame(testDataFrame)
        #
        """dc=DataCollection()
        categoricalColumns=dc.findCategorical(testDataFrame)
        testDataHex[categoricalColumns] = testDataHex[categoricalColumns].asfactor()"""
        #
        recon_error = model.anomaly(testDataHex, per_feature = True)
        # find averages of columns from the same category
        recon_error_avg  = pd.DataFrame(columns = testDataFrame.columns.values)
        for col in recon_error_avg.columns.values:
            temp_df=h2o.as_list(recon_error[[i for i in range(len(recon_error.columns)) if recon_error.columns[i].startswith('reconstr_'+col)]])
            recon_error_avg[col]=temp_df.mean(axis=1)
        recon_error_preprocessed=preprocessing.normalize(recon_error_avg, norm='l2',axis=0)
        return pd.DataFrame.from_records(recon_error_preprocessed)

