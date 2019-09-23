# lstm autoencoder to recreate a timeseries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from .PatternDiscovery import PatternDiscovery
'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

class LSTMAutoencoder(PatternDiscovery):

 def temporalize(self,timeseries, timesteps):
    #print("****timeseries****")
    #print (timeseries)
    #print (timeseries.shape)
    #n_features = timeseries.shape[1]
    X = timeseries
    y = np.zeros(len(timeseries))
    lookback = timesteps
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y


 #timeseries = genfromtxt('shuttle.csv', skip_header=1, delimiter=',', skip_footer=14400)

 def tuneAndTrain(self,timeseries):
    #timesteps = timeseries.shape[0]
    timesteps=3
    X,y=self.temporalize(timeseries, timesteps)
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
    return model


 def assignInvalidityScore(self,model, timeseries):
    # demonstrate reconstruction
    #print ("*******timeseries2********")
    #print (timeseries)
    timesteps=3
    X,y=self.temporalize(timeseries, timesteps)
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
    #print (X.shape[0])
    for i in range((X.shape[0])):
        #print("i:")
        #print(i)
        #where ax=0 is per-column, ax=1 is per-row and ax=None gives a grand total
        byRow=np.square(X[i]-yhat[i]).mean(axis=1)        
        mse_timeseries.append(np.square(X[i]-yhat[i]).mean(axis=None))
        mse_records.append(byRow)
        byRowArr=np.array([byRow])
        mse_attributes.append(np.square(X[i]-yhat[i]).mean(axis=0))
        yhatWithInvalidityScores.append(np.concatenate((yhat[i],byRowArr.T),axis=1))
        XWithInvalidityScores.append(np.concatenate((X[i],byRowArr.T),axis=1))
    print ("mse_timeseries***************"    )
    print (mse_timeseries)
    print ("mse_records*******************")
    print (mse_records)
    print ("mse_attributes****************")
    print (mse_attributes)
    return mse_timeseries, mse_records, mse_attributes, yhatWithInvalidityScores, XWithInvalidityScores



