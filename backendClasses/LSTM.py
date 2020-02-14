# lstm autoencoder to recreate a timeseries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from .PatternDiscovery import PatternDiscovery
from operator import add
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import statsmodels
import statistics

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

class LSTM(PatternDiscovery):


 def tuneAndTrain(self,timeseries):
    n_features=timeseries.shape[1]
    X = np.array(timeseries)
    X=X[0:-1,:]
    Y=X[1::,:]
    model = Sequential()
    win_size=10
    model.add(LSTM(20, activation=('tanh'), dropout=0.2, recurrent_dropout=0.2, stateful = False, input_shape = (win_size,n_features)))
    model.compile(loss=keras.losses.mean_squared_error, optimizer='rmsprop')
    model.summary()
    model.fit(X, Y,epochs=10,verbose=1)
    return model


 def assignInvalidityScore(self,model, timeseries,labels):
        X = np.array(timeseries)
        X=X[0:-1,:]
        Y=X[1::,:]
        yhat=model.predict(X)
        return np.square(Y-yhat).mean(axis=None).ravel()
     



