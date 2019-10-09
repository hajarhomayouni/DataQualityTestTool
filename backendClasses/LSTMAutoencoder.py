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

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

class LSTMAutoencoder(PatternDiscovery):

 def temporalize(self,timeseries, timesteps,features=None):
    print("****timeseries****")
    print (timeseries)
    #print (timeseries.shape)
    #n_features = timeseries.shape[1]
    X = timeseries
    y = np.zeros(len(timeseries))
    lookback = timesteps
    output_X = []
    output_y = []
    #
    dataFrameTimeseries=pd.DataFrame()
    #
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            if len(X.shape)>1:
                t.append(X[[(i+j+1)], :])
                #
                #convert the matrix to data frame
                dataFrameTemp=pd.DataFrame(data=X[[(i+j+1)], :], columns=features)
                dataFrameTemp["timeseriesId"]=i
                dataFrameTimeseries=pd.concat([dataFrameTimeseries,dataFrameTemp])
                #
            else:
                t.append(X[i+j+1])
        print ("dfTimeseries***************")
        print(dataFrameTimeseries)
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y,dataFrameTimeseries


 #timeseries = genfromtxt('shuttle.csv', skip_header=1, delimiter=',', skip_footer=14400)

 def tuneAndTrain(self,timeseries):
    #timesteps = timeseries.shape[0]
    timesteps=5
    X,y,dataFrameTimeseries=self.temporalize(timeseries.to_numpy(), timesteps,timeseries.columns.values)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], timesteps, n_features)
    # define model
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # fit model
    model.fit(X, X, epochs=300, batch_size=5, verbose=0)
    return model,dataFrameTimeseries


 def assignInvalidityScore(self,model, timeseries,labels):
    # demonstrate reconstruction
    #print ("*******timeseries2********")
    #print (timeseries)
    timesteps=5
    X,y,dataFrameTimeseries=self.temporalize(timeseries, timesteps)
    l1,l2,emptyDf=self.temporalize(labels,timesteps)
    print ("l1***********")
    print (l1)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], timesteps, n_features)
    print ("********input************")
    print(X)
    yhat = model.predict(X, verbose=0)
    print ("********output***********")
    print(yhat)
    mse_timeseries=[]
    mse_records=[]
    yhatWithInvalidityScores=[]
    XWithInvalidityScores=[]
    mse_attributes=[]
    meanOfLabels=[]
    #print (X.shape[0])
    for i in range((X.shape[0])):
        #print("i:")
        #print(i)
        #where ax=0 is per-column, ax=1 is per-row and ax=None gives a grand total
        byRow=np.square(X[i]-yhat[i]).mean(axis=1)        
        byRow=[i/sum(byRow) for i in byRow]
        mse_timeseries.append(np.square(X[i]-yhat[i]).mean(axis=None))
        meanOfLabels.append(np.mean(l1[i]))
        mse_records.append(byRow)
        byRowArr=np.array([byRow])
        mse_attributes.append(np.square(X[i]-yhat[i]).mean(axis=0))
        yhatWithInvalidityScores.append(np.concatenate((yhat[i],byRowArr.T),axis=1))
        XWithInvalidityScores.append(np.concatenate((X[i],byRowArr.T),axis=1))
    print ("mse_timeseries***************"    )
    print(meanOfLabels)
    print(mse_timeseries)
    mse_timeseries=[i/sum(mse_timeseries) for i in mse_timeseries]
    mse_timeseries=list(map(add, mse_timeseries, meanOfLabels)) 
    print (mse_timeseries)

    print ("mse_records*******************")
    #mse_records=normalize(mse_records, axis=1, norm='l1')
    print (mse_records)
    print ("mse_attributes****************")
    mse_attributes=normalize(mse_attributes, axis=1, norm='l1')
    print (mse_attributes)
    return mse_timeseries, mse_records, mse_attributes, yhatWithInvalidityScores, XWithInvalidityScores



