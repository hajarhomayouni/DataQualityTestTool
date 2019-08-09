import os
from random import *
import functools
from DataCollection import DataCollection
from PatternDiscovery import PatternDiscovery
from SklearnDecisionTree import SklearnDecisionTree
from SklearnRandomForest import SklearnRandomForest
from H2oGradientBoosting import H2oGradientBoosting
from H2oRandomForest import H2oRandomForest
from H2oKmeans import H2oKmeans
from SOM import SOM
from Testing import Testing
import h2o
import numpy as np
from Autoencoder import Autoencoder
from Pyod import Pyod
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
from Evaluation import Evaluation
import datetime


import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import statistics
import pyod
from keras.models import load_model


class DQTestToolHelper:
    
   def importData(self,db,dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath):
    dataCollection=DataCollection()
    dataFrame=dataCollection.importData(dataRecordsFilePath)
    #all the data records are clean by default
    dataFrame['status']='clean'
    dataFrame['invalidityScore']=0.0
    datasetId=dataRecordsFilePath.split("/")[-1].replace('.csv','_').replace("-","_").replace(" ","_" ) + str(randint(1,10000))
    dataFrame.to_sql('dataRecords_'+datasetId, con=db, if_exists='replace')           
    
    #initialize hyperparametrs
    hyperParameters = {'rate':[0.1], 'hiddenOpt':['50,50,50'], 'l2Opt':[1e-2], 'trainedModelFilePath':trainedModelFilePath}   
    hyperParametersDataFrame= pd.DataFrame(hyperParameters) 
    hyperParametersDataFrame.to_sql('hyperParameters_'+datasetId, con=db, if_exists='replace')
    
    #store knowFaults in database
    knownFaultsFrame=pd.DataFrame()
    if len(knownFaultsFilePath)>0:
        knownFaultsFrame=dataCollection.importData(knownFaultsFilePath)
    knownFaultsFrame.to_sql('knownFaults_'+datasetId, con=db, if_exists='replace')
    return datasetId


   """def AF(self,db,dataFrame):
       AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])
        return AFdataFrameOld"""

   def constraintDiscoveryAndFaultDetection(self,db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame):
    truePositive=0.0
    truePositiveRate=0.0
    NRDataFrame=pd.read_sql(sql="SELECT count(*) FROM scores where dataset_id like '"+datasetId+"'", con=db)
    NR=NRDataFrame[NRDataFrame.columns.values[0]].values[0]
    trainedModelFilePathDataFrame=pd.read_sql(sql="select trainedModelFilePath from hyperParameters_"+datasetId, con=db)
    trainedModelFilePath=trainedModelFilePathDataFrame[trainedModelFilePathDataFrame.columns.values[0]].values[0]

    dataCollection=DataCollection()
    #dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'status','invalidityScore'], axis=1))
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'invalidityScore'], axis=1))

    #prepare hyper-parameters
    """numberOfActualFaultsDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'actual%'",con=db)
    numberOfActualFaults=numberOfActualFaultsDataFrame[numberOfActualFaultsDataFrame.columns.values[0]].values[0]
    rateDataFrame=pd.read_sql(sql="select rate from hyperParameters_"+datasetId, con=db)
    rate=rateDataFrame[rateDataFrame.columns.values[0]].values[0]
    if numberOfActualFaults<=numberOfSuspicious:
        rate=rate*((0.1)**NR)
        if rate<0:
            rate=0.001
        db.execute("update hyperParameters set epochs="+str(rate))"""
    #prepare training data
    #select actual faluts from the current run after updating the database - we need this information to measure the false negative rate
    AFdataFrame=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults%'", con=db)

    AFdataFrame.to_sql('actualFaults_'+datasetId, con=db, if_exists='append')
    #dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where status not like 'actual%' and status not like 'invalid'", con=db)
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    
    """validDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'valid'", con=db)
    for i in range(50):
        dataFrameTrain=dataFrameTrain.append(validDataFrame, ignore_index=True)"""

    #dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0],'status','invalidityScore'], axis=1))
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0],'invalidityScore'], axis=1))
    #uncomment if you want label as a new attribute for training. At the moment y does not have any effect
    y=dataFrameTrain['status'].replace('valid',-1).replace('invalid',1).replace('actual*',1,regex=True).replace('clean',0)
    #dataFrameTrainPreprocessed['y']=y

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@Set up constraint model parameters@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    patternDiscovery = PatternDiscovery()
    bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain("","","")
    hyperParameters=[]
    n=len(dataFrameTrainPreprocessed.columns.values)
    hidden_neurons=[ n/2, n/2]
    MLmodels={"H2O_Autoencoder":H2OAutoEncoderEstimator(), "KNN": pyod.models.knn.KNN(), "SO_GAAL":pyod.models.so_gaal.SO_GAAL(),"MO_GAAL": pyod.models.mo_gaal.MO_GAAL(), "PCA": pyod.models.pca.PCA(), "MCD": pyod.models.mcd.MCD(),"OCSVM": pyod.models.ocsvm.OCSVM(), "Pyod_Autoencoder":pyod.models.auto_encoder.AutoEncoder(hidden_neurons=hidden_neurons, epochs=10,contamination=0.5,l2_regularizer=0.5, batch_size=len(dataFrameTrain)), "LOF":pyod.models.lof.LOF()}
    preTrainedModel=None
    if trainedModelFilePath!="":
        if "H2O_Autoencoder" in constraintDiscoveryMethod:
            preTrainedModel = h2o.load_model(str(trainedModelFilePath))
        """elif "Pyod_Autoencoder" in constraintDiscoveryMethod:
            preTrainedModel=load_model(str(trainedModelFilePath)+".h5")"""
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@setup new model@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if constraintDiscoveryMethod=="H2O_Autoencoder":
        patternDiscovery=Autoencoder()            
        #hiddenOpt=[[100],[100,100],[50,50],[50,50,50],[5,5,5],[20,20]]
        hiddenOpt=[[100,100]]
        l2Opt = [1e-4]
        hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
        MLmodels[constraintDiscoveryMethod]=H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=70,standardize = True,categorical_encoding='auto',export_weights_and_biases=True, quiet_mode=False,l2=1e-4, train_samples_per_iteration=-1,pretrained_autoencoder=preTrainedModel, rate=0.1, hidden=[100,100])
    else:
        patternDiscovery=Pyod()

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@Train Model@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain(hyperParameters,MLmodels[constraintDiscoveryMethod],dataFrameTrainPreprocessed)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@save the trained model for next run use@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    bestModelFileName=""
    if "H2O_Autoencoder" in constraintDiscoveryMethod:
        bestModelPath = h2o.save_model(model=bestConstraintDiscoveryModel, path="static/model/", force=True)         
        bestModelFileName=bestModelPath.split('/')[len(bestModelPath.split('/'))-1]
        trainedModelFilePath='/static/model/'+bestModelFileName
        db.execute("update hyperparameters_"+str(datasetId)+" set trainedModelFilePath='/home/hajar_homayouni_healthdatacompas/DataQualityTestTool"+trainedModelFilePath+"'")
    else:
        bestConstraintDiscoveryModel.model_.save("static/model/pyod_"+str(datasetId)+".h5")
        db.execute("update hyperparameters_"+str(datasetId)+" set trainedModelFilePath='static/model/pyod_"+str(datasetId)+"'")

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Assign invalidity scores@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Assign invalidity scores per feature
    invalidityScoresPerFeature=patternDiscovery.assignInvalidityScorePerFeature(bestConstraintDiscoveryModel, dataFramePreprocessed)
    #Assign invalidity scores per record
    #based on average of attribute's invalidity scores
    invalidityScores=patternDiscovery.assignInvalidityScore(bestConstraintDiscoveryModel, dataFramePreprocessed)
    #based on max of attribute's invalidity scores
    if constraintDiscoveryMethod=="H2O_Autoencoder":
        #invalidityScores=invalidityScoresPerFeature.max(axis=1).values.ravel()
        tempDataFrame=invalidityScoresPerFeature.max(axis=1)+y
        invalidityScores=(tempDataFrame).values.ravel()
        

    invalidityScoresWithId= pd.concat([dataFrame[dataFrame.columns[0]], pd.DataFrame(invalidityScores, columns=['invalidityScore'])], axis=1, sort=False)
    for index, row in invalidityScoresWithId.iterrows():
        db.execute("Update dataRecords_"+datasetId+" set invalidityScore="+str(row['invalidityScore'])+" where "+dataFrame.columns.values[0]+"="+str(row[dataFrame.columns.values[0]]))
    
    invalidityScoresPerFeature= pd.concat([dataFrame[dataFrame.columns[0]], invalidityScoresPerFeature], axis=1, sort=False)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@Identify Threshold@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    numberOfKnownFaultsDataFrame=pd.read_sql(sql="SELECT count(*) FROM knownFaults_"+datasetId, con=db)
    numberOfKnownFaults=numberOfKnownFaultsDataFrame[numberOfKnownFaultsDataFrame.columns.values[0]].values[0]
    faultyThreshold=np.percentile(invalidityScores,95)        
    if numberOfKnownFaults>0:
        faultyThreshold=np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))
    
    aDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    a=(aDataFrame[aDataFrame.columns.values[0]].values[0])
    bDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    b=(bDataFrame[bDataFrame.columns.values[0]].values[0])
    cDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    c=(cDataFrame[cDataFrame.columns.values[0]].values[0])
    dDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    d=(dDataFrame[dDataFrame.columns.values[0]].values[0])
    
    if b!=0 and  b>d:   
        if d>a and d<b:
            faultyThreshold=max(a,faultyThreshold)
        elif a>=d:
            faultyThreshold=min(a,np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame))))))

    normalThreshold=np.percentile(invalidityScores,50)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@Detect faulty records@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    testing=Testing()  
    faultyRecordFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where invalidityScore >="+str(faultyThreshold), con=db)    
    #Detect normal records
    normalRecordFrame=testing.detectNormalRecords(dataFrame,invalidityScores,normalThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@store different scores@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    truePositiveRate=0.0
    if not AFdataFrame.empty:
        truePositiveRate=float(len(AFdataFrame[dataFrame.columns.values[0]].unique().tolist()))/float(faultyRecordFrame.shape[0])
    falsePositiveRate=1.0-truePositiveRate
    falseNegativeRate=0.0

    AFdataFrameOldList=[str(item) for item in AFdataFrameOld[dataFrame.columns.values[0]].unique().tolist()]
    AFdataFrameList=[str(item) for item in AFdataFrame[dataFrame.columns.values[0]].unique().tolist()]
    diffDetection=len(set(AFdataFrameOldList).difference(set(AFdataFrameList))) 
    oldDetected=len(set(AFdataFrameOldList)) 
    if not AFdataFrameOld.empty:
        falseNegativeRate=float(diffDetection)/float(oldDetected)
    trueNegativeRate=1-falseNegativeRate
    #A
    A=set(suspiciousDataFrame[suspiciousDataFrame.columns.values[0]].astype(str).tolist())
    #AF
    AF=set(AFdataFrame[AFdataFrame.columns.values[0]].astype(str).unique().tolist())
    #E
    knownFaults=pd.read_sql(sql="select distinct * from knownFaults_"+datasetId,con=db)
    E=set(knownFaults[knownFaults.columns.values[0]].astype(str).tolist())
    
    PD=SD=ND=UD=0.0
    if len(A)>0:
        evaluation=Evaluation()
        PD=evaluation.previouslyDetectedFaultyRecords(A,E)
        SD=evaluation.newlyDetectedFaultyRecords(A, E, A)
        ND=evaluation.newlyDetectedFaultyRecords(A, E, AF)
        UD=evaluation.unDetectedFaultyRecords(A, E)
    db.execute('INSERT INTO scores (time, dataset_id,previously_detected,suspicious_detected,undetected,newly_detected, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate) VALUES (?,?,?,?,?, ?, ?, ?, ?,?)',(datetime.datetime.now(), datasetId,PD,SD,UD,ND,truePositiveRate, falsePositiveRate,trueNegativeRate, falseNegativeRate))
    return faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName
