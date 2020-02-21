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
import keras

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

class LSTMAutoencoder(PatternDiscovery):


 # Make a windowing fcn
 #Now the overlap is w-1, where w is the window size
 def temporalize(self,arr,win_size,step_size,features=None):
  """
  arr: any 2D array whose columns are distinct variables and 
    rows are data records at some timestamp t
  win_size: size of data window (given in data points)
  step_size: size of window step (given in data point)
  
  Note that step_size is related to window overlap (overlap = win_size - step_size), in 
  case you think in overlaps."""
  #
  dataFrameTimeseries=pd.DataFrame()
  #

  w_list = list()
  n_records = arr.shape[0]
  remainder = (n_records - win_size) % step_size 
  num_windows = 1 + int((n_records - win_size - remainder) / step_size)
  for k in range(num_windows):
    w_list.append(arr[k*step_size:win_size-1+k*step_size+1])
    #
    #convert the matrix to data frame
    dataFrameTemp=pd.DataFrame(data=arr[k*step_size:win_size-1+k*step_size+1], columns=features)
    dataFrameTemp["timeseriesId"]=k
    dataFrameTimeseries=pd.concat([dataFrameTimeseries,dataFrameTemp])
    #
  print(dataFrameTimeseries)
  return np.array(w_list),dataFrameTimeseries

 #identifies window size based on autocorrelation
 def identifyWindowSize(self,timeseries):
     dataFrameTimeseries=pd.DataFrame(timeseries)
     win_size=1
     win_sizes_of_columns=[]
     #exclude first two columns which are id and time
     for column in dataFrameTimeseries.columns.values[2:]:
         print("********column")
         print(column)
         #acf
         acf, confint=statsmodels.tsa.stattools.acf(dataFrameTimeseries[column], unbiased=False, nlags=100, qstat=False, fft=None, alpha=.05, missing='none')
         lag_ac=1
         for i in range(2,101):
             if abs(acf[i])>abs(confint[i,0]):
                 lag_ac=i
                 win_sizes_of_columns.append(i)
             else:
                 break
         #pacf
         """pacf, confint=statsmodels.tsa.stattools.pacf(dataFrameTimeseries[column], nlags=100,   alpha=0.05)
         lag_ac=1
         for i in range(2,100):
             if abs(pacf[i])>= abs(confint[i,0]):
                 lag_ac=i
             else:
                 break"""
         #
         if lag_ac>win_size:
             win_size=lag_ac

     #return (int)(statistics.mean(win_sizes_of_columns))
     return win_size




 """def temporalize(self,timeseries, timesteps,features=None):
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
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y,dataFrameTimeseries"""



 def tuneAndTrain(self,timeseries,win_size):
    #timeseries=timeseries.drop(['id','time'],axis=1)
    #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
    #K.set_session(sess)
    #win_size=min(1000,self.identifyWindowSize(timeseries))
    print("window size************")
    print(win_size)
    print("overlap*****************")
    print(int(win_size/2))
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries.to_numpy(),win_size,win_size-overlap,timeseries.columns.values)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    #print ("X**********")
    #print(X)
    # define model
    model = Sequential()
    model.add(LSTM(20,activation='relu', input_shape=(win_size,n_features-2), return_sequences=False))
    #model.add(LSTM(3, activation='relu', return_sequences=False))
    model.add(RepeatVector(win_size))
    #model.add(LSTM(3, activation='relu', return_sequences=True))
    model.add(LSTM(20, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features-2)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # fit model
    model.fit(np.delete(X,[0,1],axis=2), np.delete(X,[0,1],axis=2), epochs=5,batch_size=win_size, verbose=1)
    """print("Model Weights*******************")
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print("*****************")
        print(len(h))
        print(h)
        print("*************")"""

    return model,dataFrameTimeseries


 def assignInvalidityScore(self,model, timeseries,labels,win_size):
    #timeseries=timeseries.drop(['id','time'],axis=1)
    #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
    #K.set_session(sess)
    #win_size=min(1000,self.identifyWindowSize(timeseries))
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries,win_size,win_size-overlap)
    l1,emptyDf=self.temporalize(labels,win_size,win_size-overlap)
    #print("l1***")
    #print(l1)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    print("X**********")
    print(X)
    print(X.shape)
    yhat = model.predict(np.delete(X,[0,1],axis=2), verbose=1)
    print("yhat*************")
    print(yhat)
    mse_timeseries=[]
    mse_records=[]
    yhatWithInvalidityScores=[]
    XWithInvalidityScores=[]
    mse_attributes=[]
    maxOfLabels=[]
    for i in range((X.shape[0])):
        #where ax=0 is per-column, ax=1 is per-row and ax=None gives a grand total
        XWithoutIdAndTime=np.delete(X,[0,1],axis=2)
        #print("XWithoutIdAndTime")
        #print(XWithoutIdAndTime)
        byRow=np.square(XWithoutIdAndTime[i]-yhat[i]).mean(axis=1)        
        byRow=[i/sum(byRow) for i in byRow]
        mse_timeseries.append(np.square(XWithoutIdAndTime[i]-yhat[i]).mean(axis=None))
        maxOfLabels.append(np.max(l1[i]))
        mse_records.append(byRow)
        byRowArr=np.array([byRow])
        mse_attributes.append(np.square(XWithoutIdAndTime[i]-yhat[i]).mean(axis=0))
        yhatWithInvalidityScores.append(np.concatenate((yhat[i],byRowArr.T),axis=1))
        XWithInvalidityScores.append(np.concatenate((X[i],byRowArr.T),axis=1))
    #print ("mse_timeseries***************"    )
    mse_timeseries=[i/sum(mse_timeseries) for i in mse_timeseries]
    mse_timeseries=list(map(add, mse_timeseries, maxOfLabels)) 
    #print (mse_timeseries)

    #print ("mse_records*******************")
    #mse_records=normalize(mse_records, axis=1, norm='l1')
    #print (mse_attributes)
    #print ("mse_attributes****************")
    mse_attributes=normalize(mse_attributes, axis=0, norm='l1')
    return mse_timeseries, mse_records, mse_attributes, yhatWithInvalidityScores, XWithInvalidityScores



