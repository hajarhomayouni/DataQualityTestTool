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
import scipy
from scipy import signal
from eli5.sklearn import PermutationImportance

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

class LSTMAutoencoder(PatternDiscovery):

 def freq_peaks(self,sig):
  """
  Frequency estimation from find peaks method
  sig - input signal
    
  return: 
  dominant period
  """
  # Find the indices where there's a crossing
  #indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
  #crossings=np.where(np.diff(np.sign(sig)))[0]
  positive = sig > 0
  peaks=scipy.signal.find_peaks(sig)[0]

  # Let's calculate the real crossings by interpolate
  #crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
  # Let's get the time between each crossing
  # the diff function will get how many samples between each crossing
  # we divide the sampling rate to get the time between them
    
  delta_t = np.diff(peaks)
    
  # Get the mean value for the period
  period = int(np.max(delta_t))
    
  return period
 def freq_zero_crossing(self,sig, fs=1):
  """
  Frequency estimation from zero crossing method
  sig - input signal
  fs - sampling rate
    
  return: 
  dominant period
  """
  # Find the indices where there's a crossing
  positive = sig > 0
  crossings=np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
  
  # Let's get the time between each crossing
  # the diff function will get how many samples between each crossing
  # we divide the sampling rate to get the time between them
  delta_t = np.diff(crossings) / fs
    
  # Get the mean value for the period
  period =int(np.max(delta_t))*2
    
  return period

 def difference(self,dataset):
  #remove linear trend
  diff=dataset.diff(periods=-1)
  #find period length to initialize interval
  interval=self.freq_zero_crossing(dataset)
  #interval=self.freq_peaks(dataset.to_numpy().ravel())
  #remove seasonality
  diff=diff.diff(periods=-interval)
  print("interval*******")
  print(interval)
  return diff, interval

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
  return np.array(w_list),dataFrameTimeseries

 #identifies window size based on autocorrelation
 def identifyWindowSize(self,timeseries):
     dataFrameTimeseries=pd.DataFrame(timeseries)
     win_size=1
     win_sizes_of_columns=[]
     #exclude first two columns which are id and time
     for column in dataFrameTimeseries.columns.values[2:]:
         acf, confint=statsmodels.tsa.stattools.acf(dataFrameTimeseries[column], unbiased=False, nlags=100, qstat=False, fft=None, alpha=.05, missing='none')
         lag_ac=1
         for i in range(2,101):
             if abs(acf[i])>abs(confint[i,0]):
                 lag_ac=i
                 win_sizes_of_columns.append(i)
             else:
                 break
         #
         if lag_ac>win_size:
             win_size=lag_ac

     #return (int)(statistics.mean(win_sizes_of_columns))
     return win_size

 
   
    


 def tuneAndTrain(self,timeseries,win_size):
    #difference transform
    """diff, interval= self.difference(timeseries.drop([timeseries.columns.values[0], timeseries.columns.values[1]], axis=1))
    timeseries=pd.concat([timeseries[[timeseries.columns.values[0], timeseries.columns.values[1]]],diff], axis=1)
    timeseries=timeseries.head(len(diff)-(interval+1))"""
    #
    print("window size************")
    print(win_size)
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
    """def custom_loss(y_true,y_pred):
        return K.max(K.square(y_pred - y_true))"""
    model.compile(optimizer='adam', loss="mse")
    model.summary()
    # fit model
    model.fit(np.delete(X,[0,1],axis=2), np.delete(X,[0,1],axis=2), epochs=100,batch_size=win_size, verbose=1)
    """print("Model Weights*******************")
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print("*****************")
        print(len(h))
        print(h)
        print("*************")"""
    """#show feature importance
    perm = PermutationImportance(model, random_state=1, scoring='f1_micro').fit(X,X)
    eli5.show_weights(perm, feature_names = timeseries.columns.vaues[2:])"""
    return model,dataFrameTimeseries


 def assignInvalidityScore(self,model, timeseries,labels,win_size):
    #difference transform
    """diff, interval= self.difference(timeseries.drop([timeseries.columns.values[0], timeseries.columns.values[1]], axis=1))
    timeseries=pd.concat([timeseries[[timeseries.columns.values[0], timeseries.columns.values[1]]],diff], axis=1)
    timeseries=timeseries.head(len(diff)-(interval+1))"""
    #
    timeseries=timeseries.to_numpy()
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries,win_size,win_size-overlap)
    l1,emptyDf=self.temporalize(labels,win_size,win_size-overlap)
    #print("l1***")
    #print(l1.shape)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    #print("X**********")
    #print(X)
    #print(X.shape)
    yhat = model.predict(np.delete(X,[0,1],axis=2), verbose=1)
    """print("test***************")
    test=np.delete(X,[0,1],axis=2)[0:1]
    print(test)
    print(model.predict(test,verbose=1))"""
    #print("yhat*************")
    #print(yhat.shape)
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
        mse_timeseries.append(np.square(XWithoutIdAndTime[i]-yhat[i]).max(axis=None))
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

    #mse_attributes=normalize(mse_attributes, axis=0, norm='l1')
    
    #find_LSbs_approach 4:
    print("LSBs************************")
    print(mse_attributes)
    #print(mse_attributes.shape)

    print("#############################")
    print(np.mean(mse_attributes, axis=0))
    #
    return mse_timeseries, mse_records, mse_attributes, yhatWithInvalidityScores, XWithInvalidityScores

    
 def findLsbs_1(self,model, timeseries,win_size):
    #This method finds LSbs based on randomizing LSBs together
    #difference transform
    """diff, interval= self.difference(timeseries.drop([timeseries.columns.values[0], timeseries.columns.values[1]], axis=1))
    timeseries=pd.concat([timeseries[[timeseries.columns.values[0], timeseries.columns.values[1]]],diff], axis=1)
    timeseries=timeseries.head(len(diff)-(interval+1))"""
    #
    timeseries=timeseries.to_numpy()
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries,win_size,win_size-overlap)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    XWithoutIdAndTime=np.delete(X,[0,1],axis=2)

    testArray=XWithoutIdAndTime
    MSE=[]
    for i in range(1,7):
        pos=8-i
        random_binary_matrix = np.random.randint(0,2,size=(testArray.shape[0],testArray.shape[1],7-pos+1))
        print("random_binary_matrix")
        print(random_binary_matrix)
        testArray[:,:,pos:8]=random_binary_matrix
        print("testArray******")
        print(testArray)
        print("XwithoutId and time")
        print(XWithoutIdAndTime)
        
        yhat = model.predict(testArray, verbose=1)
        loss = model.evaluate(testArray, testArray, verbose=1)
        print("yhat******")
        print(yhat)

        mse=np.square(testArray-yhat).mean(axis=1)        
        mse=np.mean(mse)
        MSE.append(mse)

    print("mse*************************************")
    print(MSE)
    return MSE

 def findLsbs_2(self,model, timeseries,win_size):
    #This method finds LSbs based on randomizing LSBs one by one
    #difference transform
    """diff, interval= self.difference(timeseries.drop([timeseries.columns.values[0], timeseries.columns.values[1]], axis=1))
    timeseries=pd.concat([timeseries[[timeseries.columns.values[0], timeseries.columns.values[1]]],diff], axis=1)
    timeseries=timeseries.head(len(diff)-(interval+1))"""
    #
    timeseries=timeseries.to_numpy()
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries,win_size,win_size-overlap)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    XWithoutIdAndTime=np.delete(X,[0,1],axis=2)

    MSE=[]
    for i in range(1,7):
        testArray=XWithoutIdAndTime
        pos=8-i
        random_binary_matrix = np.random.randint(0,2,size=(testArray.shape[0],testArray.shape[1],1))
        print("random_binary_matrix")
        print(random_binary_matrix)
        testArray[:,:,pos:pos+1]=random_binary_matrix
        print("testArray******")
        print(testArray)
        print("XwithoutId and time")
        print(XWithoutIdAndTime)
        
        yhat = model.predict(testArray, verbose=1)
        loss = model.evaluate(testArray, testArray, verbose=1)
        print("yhat******")
        print(yhat)

        mse=np.square(testArray-yhat).mean(axis=1)        
        mse=np.mean(mse)
        MSE.append(mse)

    print("mse*************************************")
    print(MSE)
    return MSE

 def findLsbs_3(self,model, timeseries,win_size):
    #This method finds LSbs based on randomizing all bits one by one
    #difference transform
    """diff, interval= self.difference(timeseries.drop([timeseries.columns.values[0], timeseries.columns.values[1]], axis=1))
    timeseries=pd.concat([timeseries[[timeseries.columns.values[0], timeseries.columns.values[1]]],diff], axis=1)
    timeseries=timeseries.head(len(diff)-(interval+1))"""
    #
    timeseries=timeseries.to_numpy()
    overlap=1#int(win_size/2)
    X,dataFrameTimeseries=self.temporalize(timeseries,win_size,win_size-overlap)
    n_features=timeseries.shape[1]
    X = np.array(X)
    X = X.reshape(X.shape[0], win_size, n_features)
    XWithoutIdAndTime=np.delete(X,[0,1],axis=2)

    MSE=[]
    for i in range(0,16):
        testArray=XWithoutIdAndTime
        pos=i
        random_binary_matrix = np.random.randint(0,2,size=(testArray.shape[0],testArray.shape[1],1))
        print("random_binary_matrix")
        print(random_binary_matrix)
        testArray[:,:,pos:pos+1]=random_binary_matrix
        print("testArray******")
        print(testArray)
        print("XwithoutId and time")
        print(XWithoutIdAndTime)
        
        yhat = model.predict(testArray, verbose=1)
        loss = model.evaluate(testArray, testArray, verbose=1)
        print("yhat******")
        print(yhat)

        mse=np.square(testArray-yhat).mean(axis=1)        
        mse=np.mean(mse)
        MSE.append(mse)

    print("mse*************************************")
    print(MSE)
    return MSE


