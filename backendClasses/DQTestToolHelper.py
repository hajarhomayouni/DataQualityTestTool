import os
from random import *
import functools
from .DataCollection import DataCollection
from .PatternDiscovery import PatternDiscovery
from .SklearnDecisionTree import SklearnDecisionTree
from .SklearnRandomForest import SklearnRandomForest
from .H2oGradientBoosting import H2oGradientBoosting
from .H2oRandomForest import H2oRandomForest
from .H2oKmeans import H2oKmeans
from .SOM import SOM
from .Testing import Testing
import h2o
import numpy as np
from .Autoencoder import Autoencoder
from .LSTMAutoencoder import LSTMAutoencoder
from .LSTM import LSTM
from .Pyod import Pyod
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
from .Evaluation import Evaluation
from .TSFeatures import TSFeatures
import datetime


import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import statistics
import pyod
from keras.models import load_model
from tsfresh import select_features,extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
import re
import decimal
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


class DQTestToolHelper:
   
   def highlight_greaterthan(self,s,threshold,column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]

   
   def importData(self,db,dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath):
    dataCollection=DataCollection()
    dataFrame=dataCollection.importData(dataRecordsFilePath)#.head(150)
    #dataFrame=dataFrame[['id','time','B0','B1']]
    dataFrame['status']='clean'
    dataFrame['invalidityScore']=0.0
    datasetId=dataRecordsFilePath.split("/")[-1].replace('.csv','_').replace("-","_").replace(" ","_" ) + str(randint(1,10000))
    dataFrame.to_sql('dataRecords_'+datasetId, con=db, if_exists='replace')           
    
    #store knowFaults in database
    knownFaultsFrame=pd.DataFrame()
    if len(knownFaultsFilePath)>0:
        knownFaultsFrame=dataCollection.importData(knownFaultsFilePath)
    knownFaultsFrame.to_sql('knownFaults_'+datasetId, con=db, if_exists='replace')
    return datasetId


   def constraintDiscoveryAndFaultDetection(self,db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,hyperParameters,grouping_attr=None,TP_T=None,win_size=None):
    truePositive=0.0
    truePositiveRate=0.0
    NRDataFrame=pd.read_sql(sql="SELECT count(*) FROM scores where dataset_id like '"+datasetId+"'", con=db)
    NR=NRDataFrame[NRDataFrame.columns.values[0]].values[0]

    dataCollection=DataCollection()
    dataFramePreprocessed=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop(['invalidityScore','status'], axis=1),grouping_attr)
    else:
        dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'invalidityScore','status'], axis=1))

    #y=dataFrame['status'].replace('valid',-1).replace('invalid',1).replace('actual*',1,regex=True).replace('clean',0).replace('suspicious*',0,regex=True)
    #dataFramePreprocessed['status']=y

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #prepare training data@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #select actual faluts from the current run after updating the database - we need this information to measure the false negative rate
    AFdataFrame=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults%'", con=db)

    AFdataFrame.to_sql('actualFaults_'+datasetId, con=db, if_exists='append')
    #dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where status not like 'actual%' and status not like 'invalid'", con=db)
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    
    dataFrameTrainPreprocessed=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop(['invalidityScore','status'], axis=1),grouping_attr)
    else:
        dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0],'invalidityScore','status'], axis=1))
        #replace suspicious with zero only for tuning parameters via testScriptTuning.py (we do not want any feedback in that case)
    #suspicious is zero because we are deciding based on the expert idea in previous time not based on our idea
    y=dataFrameTrain['status'].replace('valid',-1).replace('invalid',1).replace('actual*',1,regex=True).replace('clean',0).replace('suspicious*',0,regex=True)
    #dataFrameTrainPreprocessed['status']=y
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@Train Model@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    dataFrameTimeseries=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        patternDiscovery=LSTMAutoencoder()
        if win_size==None:
            win_size=patternDiscovery.identifyWindowSize(dataFramePreprocessed,grouping_attr)
            hyperParameters={"auto_win_size:"+str(win_size)}
        bestConstraintDiscoveryModel,dataFrameTimeseries=patternDiscovery.tuneAndTrain(dataFrameTrainPreprocessed,win_size,grouping_attr)
    elif constraintDiscoveryMethod=="H2O_Autoencoder":
        patternDiscovery=Autoencoder()           
        model=H2OAutoEncoderEstimator(activation='Tanh', epochs=hyperParameters['epochs'],export_weights_and_biases=True, quiet_mode=False,hidden=hyperParameters['hidden'], categorical_encoding="auto", standardize=True)#,hidden_dropout_ratios=hyperParameters['hidden_dropout_ratios'], input_dropout_ratio=hyperParameters['input_dropout_ratio'],l2=hyperParameters['l2'])
        bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain(model,dataFrameTrainPreprocessed)
    elif constraintDiscoveryMethod=="LSTM":
        patternDiscovery=LSTM()
        bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain(dataFramePreprocessed)
    

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Assign invalidity scores@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    invalidityScores=[]
    invalidityScoresPerFeature=[]
    yhatWithInvalidityScores=[]
    XWithInvalidityScores=[]
    XRawWithInvalidityScores=[]
    mse_attributes=[]
    networkError=0.0
    if "LSTMAutoencoder" in constraintDiscoveryMethod:
        timeseries_raw=dataFrame.drop(['invalidityScore','status'], axis=1)
        timeseries=dataFramePreprocessed
        mse_timeseries, mse_records, mse_attributes,yhatWithInvalidityScores,XWithInvalidityScores,XRawWithInvalidityScores=patternDiscovery.assignInvalidityScore(bestConstraintDiscoveryModel,timeseries,timeseries_raw,y,win_size,grouping_attr)
        invalidityScores=mse_timeseries
        #patternDiscovery.findLsbs_3(bestConstraintDiscoveryModel,dataFramePreprocessed,win_size)
    elif constraintDiscoveryMethod=="LSTM":
        inalidityScores=patternDiscovery.assignInvalidityScore(bestConstraintDiscoveryModel, dataFramePreprocessed)
    else:
        #Assign invalidity scores per feature
        invalidityScoresPerFeature=patternDiscovery.assignInvalidityScorePerFeature(bestConstraintDiscoveryModel, dataFramePreprocessed)
        #Assign invalidity scores per record
        #based on average of attribute's invalidity scores
        invalidityScores=patternDiscovery.assignInvalidityScore(bestConstraintDiscoveryModel, dataFramePreprocessed)
        networkError=bestConstraintDiscoveryModel.mse()
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
    faultyThreshold=np.percentile(invalidityScores,98)        
    #
    X=np.array(XRawWithInvalidityScores)
    temp=X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    cols=dataFrame.columns.values[:-2]
    cols=np.append(cols,'invalidityScore')
    #TODO: remove duplicate records based on value of IDs (after merging timeseris into one 2D array)
    invalidityScoresPerRecord=pd.DataFrame(temp, columns=cols)['invalidityScore']
    faultyThresholdRecords=np.percentile(invalidityScoresPerRecord,85)    
    #
    E=set()
    knownFaults=pd.read_sql(sql="select distinct * from knownFaults_"+datasetId,con=db)
    if not knownFaults.empty:
        E=set(knownFaults[knownFaults.columns.values[0]].astype(str).tolist())
    #NumRealFaultyTimeseries
    P_T=0.0
    for timeseriesIndex in range(len(invalidityScores)):
        timeseries=dataFrameTimeseries[dataFrameTimeseries['timeseriesId'] == timeseriesIndex]
        if len(set(timeseries[dataFrame.columns.values[0]].astype(int).astype(str).tolist()).intersection(E))>0:
                P_T+=1
    print("P_T")
    print(P_T)
    #
    if numberOfKnownFaults>0:
        if constraintDiscoveryMethod=="H2O_Autoencoder" or constraintDiscoveryMethod=="LSTM":
            faultyThreshold=np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))
        elif constraintDiscoveryMethod=="LSTMAutoencoder":
            faultyThresholdRecords=np.percentile(invalidityScoresPerRecord, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))
            totalNumOfTimeseries=len(dataFrame)/win_size
            numOfGroupsToBeReported=P_T
            if P_T==0:
                minOfNumOfFaultyGroups=numberOfKnownFaults/win_size
                maxOfNumOfFaultyGroups=numberOfKnownFaults
                meanOfNumOfFaultyGroups=(minOfNumOfFaultyGroups+maxOfNumOfFaultyGroups)/2
                numOfGroupsToBeReported=minOfNumOfFaultyGroups+1
                if numOfGroupsToBeReported>=totalNumOfTimeseries:
                    numOfGroupsToBeReported=minOfNumOfFaultyGroups+1
                elif numberOfKnownFaults<win_size:
                    numOfGroupsToBeReported=3
            faultyThreshold=np.percentile(invalidityScores, 100-100*(numOfGroupsToBeReported/totalNumOfTimeseries))


    aDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    a=(aDataFrame[aDataFrame.columns.values[0]].values[0])
    bDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    b=(bDataFrame[bDataFrame.columns.values[0]].values[0])
    cDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    c=(cDataFrame[cDataFrame.columns.values[0]].values[0])
    dDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    d=(dDataFrame[dDataFrame.columns.values[0]].values[0])
    
    #print("%%%%%%%%%%%%%")
    #print("a:"+str(a))
    #print("b:"+str(b))
    #print("c:"+str(c))
    #print("d:"+str(d))

    if b:
        if b!=0 and  b>d:   
            if d>a and d<b:
                faultyThreshold=max(a,faultyThreshold)
            elif a>=d:
                if constraintDiscoveryMethod=="LSTMAutoencoder":
                    #faultyThreshold=max(0,min(a,np.percentile(invalidityScores, 98-(100*(float(numberOfKnownFaults/win_size)/float(len(dataFrame)/win_size))))))
                    #faultyThreshold=max(0,min(a,np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults*win_size)/float(len(dataFrameTimeseries)))/2))))
                    faultyThreshold=max(0, min(a,np.percentile(invalidityScores, 100-100*(numberOfGroupsToBeReported/totalNumOfTimeseries))))
                if constraintDiscoveryMethod=="H2O_Autoencoder" or constraintDiscoveryMethod=="LSTM":
                    faultyThreshold=max(0,min(a,np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))))


    normalThreshold=np.percentile(invalidityScores,50)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@Detect faulty records@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    faultyTimeseriesIndexes=[]
    normalTimeseriesIndexes=[]
    testing=Testing()  
    faultyRecordFrame=pd.DataFrame()
    normalRecordFrame=pd.DataFrame()
    if "LSTMAutoencoder" in constraintDiscoveryMethod:
        faultyTimeseriesIndexes=np.where(invalidityScores>faultyThreshold)
        normalTimeseriesIndexes=np.where(invalidityScores<=faultyThreshold)
        faultyRecordsInTimeseries=pd.DataFrame()
        normalRecordsInTimeseries=pd.DataFrame()
        for i in faultyTimeseriesIndexes[0]:
            faultyRecordsInTimeseries=pd.concat([faultyRecordsInTimeseries, pd.DataFrame(XRawWithInvalidityScores[i][:,0])])
        for j in normalTimeseriesIndexes[0]:
            normalRecordsInTimeseries=pd.concat([normalRecordsInTimeseries, pd.DataFrame(XRawWithInvalidityScores[i][:,0])])
        if not faultyRecordsInTimeseries.empty:
            faultyRecordsInTimeseries.to_sql("faultyRecords_temp_"+datasetId, con=db, if_exists='replace', index=False)
            faultyRecordFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where "+dataFrame.columns.values[0]+ " IN (Select * FROM faultyRecords_temp_"+datasetId+" )",con=db)
            db.execute("Drop table faultyRecords_temp_"+datasetId)
    else:
        faultyRecordFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where invalidityScore >="+str(faultyThreshold), con=db)    
        #Detect normal records
        normalRecordFrame=testing.detectNormalRecords(dataFrame,invalidityScores,normalThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@store different scores@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ########################################
    #Calculate scores at record level#######
    #TODO: re-calc the record-level scores based on the confusion table
    ########################################
    TPR=0.0
    AFdataFrameList=[]
    AFdataFrameOldList=[]
    A=set()
    AF=set()
    #E=set()
    if not AFdataFrame.empty:
        AFdataFrameList=[str(item) for item in AFdataFrame[dataFrame.columns.values[0]].unique().tolist()]
        TPR=float(len(AFdataFrame[dataFrame.columns.values[0]].unique().tolist()))/float(faultyRecordFrame.shape[0])
        #AF
        AF=set(AFdataFrame[AFdataFrame.columns.values[0]].astype(str).unique().tolist())
    if not AFdataFrameOld.empty:
        AFdataFrameOldList=[str(item) for item in AFdataFrameOld[dataFrame.columns.values[0]].unique().tolist()]
    FPR=1.0-truePositiveRate
    FNR=0.0
    diffDetection=len(set(AFdataFrameOldList).difference(set(AFdataFrameList))) 
    oldDetected=len(set(AFdataFrameOldList)) 
    if not AFdataFrameOld.empty:
        FNR=float(diffDetection)/float(oldDetected)
    TNR=1-FNR
    
    #A=set(suspiciousDataFrame[suspiciousDataFrame.columns.values[0]].astype(str).tolist())
    A=set(faultyRecordFrame[faultyRecordFrame.columns.values[0]].astype(str).tolist())
    
    
    PD=SD=ND=UD=F1=0.0
    if len(A)>0:
        evaluation=Evaluation()
        PD=evaluation.previouslyDetectedFaultyRecords(A,E)
        SD=evaluation.newlyDetectedFaultyRecords(A, E, A)#*(1/win_size)
        ND=evaluation.newlyDetectedFaultyRecords(A, E, AF)
        UD=evaluation.unDetectedFaultyRecords(A, E)
    if TPR>0:
        precision=TPR/(TPR+FPR)
        recall=TPR/(TPR+FNR)
        F1=(2*precision*recall)/(precision+recall)
    ############################################
    #Calculate scores at group/time series level
    ############################################
    #NumRealFaultyTimeseries
    #P_T=0.0
    #NumRealNormalTimeseries
    N_T=0.0
    #Number of actual faulty records
    P=float(len(E))
    #Number of detected faulty records 
    a=set()
    #Number of actual valid records
    N=float(len(dataFrame))-P
    #set of actual valid records
    n=set(dataFrame[dataFrame.columns.values[0]].astype(int).astype(str).tolist()).difference(E)
    F1_T=FP_T=FPR_T=TPR_T=0.0
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        for timeseriesIndex in faultyTimeseriesIndexes[0]:
                temp=XRawWithInvalidityScores[timeseriesIndex]
                cols=dataFrame.columns.values[0:-2]
                cols=np.append(cols,'invalidityScore')
                tempdf=pd.DataFrame(temp, columns=cols)
                #Based on a threshold on overall invalidity scores per records
                #a=a.union(set(tempdf.loc[(tempdf['invalidityScore'] >= faultyThresholdRecords)][dataFrame.columns.values[0]].astype(int).astype(str).tolist()))
                #Based on mean of invalidity scores in each subsequence
                a=a.union(set(tempdf.loc[(tempdf['invalidityScore'] >= tempdf['invalidityScore'].mean())][dataFrame.columns.values[0]].astype(int).astype(str).tolist()))
        print("P_T")
        print(P_T)

        
        #calculate scores at record level for timeseries      
        TP=float(len(a.intersection(E)))
        FP=float(len(a.intersection(n)))

        if P>0:
            TPR=TP/P

        if TPR>0:
            precision=TP/(TP+FP)
            recall=TPR
            F1=(2*precision*recall)/(precision+recall)
        #


        N_T=float(len(invalidityScores))-P_T
        #if None means if it is the first time we run the tool or if we are in command line mode
        if TP_T is None:
            TP_T=0.0
            for i in faultyTimeseriesIndexes[0]:
                (dataFrameTimeseries.loc[dataFrameTimeseries['timeseriesId'] == i]).to_sql('faultyTimeseries_i', con=db, if_exists='replace', index=False)
                faultyRecordsInTimeseries_i=pd.read_sql(sql="select * from faultyTimeseries_i join knownFaults_"+datasetId+ " on faultyTimeseries_i."+dataFrame.columns.values[0]+"=knownFaults_"+datasetId+"."+dataFrame.columns.values[0], con=db)
                if (len(faultyRecordsInTimeseries_i)>0):
                    TP_T+=1.0
                else:
                    FP_T+=1.0
                db.execute("Drop table faultyTimeseries_i")

        if P_T>0:
            TPR_T=TP_T/P_T

    if TPR_T>0:
        precision_T=TP_T/(TP_T+FP_T)
        recall_T=TPR_T
        F1_T=(2*precision_T*recall_T)/(precision_T+recall_T)
    db.execute('INSERT INTO scores (time, dataset_id,HP,Loss,PD,SD,F1,UD,ND,TPR,FPR,TPR_T,FPR_T,F1_T) VALUES (?,?,?,?,?,?,?, ?, ?, ?, ?,?,?,?)',(datetime.datetime.now(), datasetId, str(hyperParameters), networkError, PD,SD,F1,UD,ND,TPR, FPR,TPR_T,FPR_T,F1_T))
    return faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,faultyThresholdRecords,yhatWithInvalidityScores,XWithInvalidityScores,XRawWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y


   def faultyTimeseriesInterpretation(self,db,dataFrame,interpretationMethod,datasetId,dataFramePreprocessed,yhatWithInvalidityScores,XWithInvalidityScores,XRawWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFrameTimeseries,y,invalidityScores,faultyThresholdRecords,grouping_attr=None):
    dataCollection=DataCollection() 
    numberOfClusters=len(faultyTimeseriesIndexes[0])
    faulty_records_html=[]
    suspicious_records_html=[]
    faulty_cluster_scores_fig_url=[]
    suspicious_cluster_scores_fig_url=[]
    faulty_timeseries_fig_url=[]
    suspicious_timeseries_fig_url=[]
    db.execute("Update dataRecords_"+datasetId+" set status='invalid' where status like 'actual%' ")   
    ##############################################################
    axis=[]
    tsFeatures=TSFeatures()
    normalFeatures=pd.DataFrame()
    for i in normalTimeseriesIndexes[0]:
        ###########################################################
        #Prepapre multicolor valid data for data visualization plot
        ###########################################################
        normalFrameTimeseries=dataFrameTimeseries.loc[dataFrameTimeseries['timeseriesId']==i]
        normalFrame=pd.merge(dataFrame,normalFrameTimeseries[dataFrame.columns.values[0]],how='inner',on=dataFrame.columns.values[0])
        axis.append(normalFrame)
        ##########################################################
        #Extract timeseries features from normal subsequences
        #########################################################
        partialNormalFrame=dataFrameTimeseries.loc[dataFrameTimeseries['timeseriesId']==i].drop([dataFrameTimeseries.columns.values[0],'time','timeseriesId'],axis=1)
        partialNormalFeatures=tsFeatures.extract_features(partialNormalFrame)
        df_attributes=pd.DataFrame()
        #exclude last index which is timeseriesId
        index=0
        for attribute_features in partialNormalFeatures:
            cols=[partialNormalFrame.columns.values[index]+"_"+col for col in ['mean','variance','lumpiness','lshift','vchange','linearity','curvature','spikiness','BurstinessFF','minimum','maximum','rmeaniqmean','moment3','highlowmu']]
            data=np.array(attribute_features)
            df_attribute=pd.DataFrame(data=data[None],columns=cols)
            df_attributes= pd.concat([df_attributes, df_attribute], axis=1)
            index+=1
        df_attributes.insert(loc=len(df_attributes.columns), column='label', value=0)
        df_attributes.insert(loc=len(df_attributes.columns), column='timeseriesId', value=i)
        normalFeatures=pd.concat([normalFeatures,df_attributes])
    normalFrame=normalFeatures
    normalFrame=normalFrame.fillna(0)
    normalFrameGeneral=normalFrame.replace(np.inf, 0)
    #######################
    faulty_cluster_dt_url=[]
    suspicious_cluster_dt_url=[]
    index=0
    for i in faultyTimeseriesIndexes[0]:
        #########################################################
        #Prepare X and Y for s-score per attribute plot
        #########################################################
        df = pd.DataFrame(XRawWithInvalidityScores[i], columns=np.append(dataFrame.columns.values[:-2],'invalidityScore'))
        X=list(dataFramePreprocessed.columns.values[2:])
        if grouping_attr:
            X.remove(grouping_attr)
       
        X=np.array(X)
        Y=mse_attributes[i]
        ########################################################
        #1. Only use faulty attributes to generate the trees
        ########################################################
        faulty_attributes=X
        #if it is a multivariate timeseries
        if len(dataFrameTimeseries.columns.values)>4:
            faulty_attributes_indexes=[i for i,v in enumerate(Y) if v > np.percentile(Y,90)]
            faulty_attributes=X[faulty_attributes_indexes]
        ########################################################
        mask,categoricalColumns=dataCollection.find_categorical(df.drop([df.columns.values[0],'invalidityScore',grouping_attr],axis=1))

        if len(categoricalColumns)>0:
            completedf=pd.DataFrame(Y.reshape(-1, len(Y)),columns=X)
            categoricaldf=pd.DataFrame()
            categoricalColumnsEncoded=[]
            for col in categoricalColumns:
                with_col = [x for x in completedf.columns.values if x.startswith(col)] 
                categoricalColumnsEncoded.append(with_col)
                tempdf=completedf[with_col]
                tempdf_mean=pd.DataFrame(tempdf.mean(axis = 1), columns=[col]) 
                categoricaldf= pd.concat([categoricaldf, tempdf_mean], axis=1)
            nonCategoricaldf=completedf.drop(list(np.concatenate(categoricalColumnsEncoded).flat), axis=1)
            finaldf=pd.concat([nonCategoricaldf,categoricaldf],axis=1)
            X=finaldf.columns.values
            Y=finaldf.to_numpy()[0]
        ##############################################################
        #Prepare actual, predicted vs time for data_visualization plot
        ##############################################################
        faulty_attribute_index=np.where(Y==max(Y))
        faulty_attribute=X[faulty_attribute_index]
        #v2_axis=np.array(yhatWithInvalidityScores)[:,:,faulty_attribute_index].flatten()
        v_title=faulty_attribute
        v1_red=dataFrameTimeseries.loc[dataFrameTimeseries['timeseriesId'] == i]#[faulty_attribute]
        left=v1_red.add_suffix('_x')
        right=dataFrame.add_suffix('_y')
        v1_merge=pd.merge(left,right, left_on=dataFrame.columns.values[0]+'_x',right_on=dataFrame.columns.values[0]+'_y',how='left')#[faulty_attribute]
        temp_str=faulty_attribute[0]+'_y'
        v1_red=v1_merge[[temp_str]]
        h_red=v1_merge["time"+"_x"]
        ################################################
        #Update status of suspicious groups in database@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        df.to_sql('suspicious_i_temp_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious_"+str(i)+ "' where  "+dataFramePreprocessed.columns.values[0]+" in (select "+dataFramePreprocessed.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")
        db.execute("Drop table suspicious_i_temp_"+datasetId)
        index=index+1
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #Add Decision Tree for each Timesereis
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        decisionTreeImageUrls=[]
        faultyFrame=dataFrameTimeseries.loc[dataFrameTimeseries['timeseriesId']==i].drop([dataFrameTimeseries.columns.values[0],'time','timeseriesId'],axis=1)
        df_attributes=pd.DataFrame()
        faultyFeatures=tsFeatures.extract_features(faultyFrame)
        attribute_index=0
        for attribute_features in faultyFeatures:
            cols=[faultyFrame.columns.values[attribute_index]+"_"+col for col in ['mean','variance','lumpiness','lshift','vchange','linearity','curvature','spikiness','BurstinessFF','minimum','maximum','rmeaniqmean','moment3','highlowmu']]
            data=np.array(attribute_features)
            df_attribute=pd.DataFrame(data=data[None],columns=cols)
            df_attributes= pd.concat([df_attributes, df_attribute], axis=1)
            attribute_index+=1
        df_attributes.insert(loc=len(df_attributes.columns), column='label', value=1)
        df_attributes.insert(loc=len(df_attributes.columns), column='timeseriesId', value=i)
        faultyFrame=df_attributes
        faultyFrame=faultyFrame.fillna(0)
        faultyFrame=faultyFrame.replace(np.inf, 0)
        ########################################################
        #2. Only use faulty attributes to generate the trees
        ########################################################
        if grouping_attr:
            faultyFrame=faultyFrame[[col for col in faultyFrame if (col.startswith(tuple(faulty_attributes)) or col in ["label","timeseriesId"]) and (not col.startswith(grouping_attr)) ]]           
            normalFrame=normalFrameGeneral[[col for col in normalFrameGeneral if (col.startswith(tuple(faulty_attributes)) or col in ["label","timeseriesId"]) and (not col.startswith(grouping_attr))]]
        else:
            faultyFrame=faultyFrame[[col for col in faultyFrame if (col.startswith(tuple(faulty_attributes)) or col in ["label","timeseriesId"]) ]]#and (not col.startswith(grouping_attr)) ]]           
            normalFrame=normalFrameGeneral[[col for col in normalFrameGeneral if (col.startswith(tuple(faulty_attributes)) or col in ["label","timeseriesId"]) ]]#and (not col.startswith(grouping_attr))]]

        ###############################################
        decisionTreeTrainingFrame=pd.concat([normalFrame,faultyFrame]).drop(['timeseriesId'],axis=1)
        decisionTreeTrainingFramePreprocessed=decisionTreeTrainingFrame
        tree=H2oGradientBoosting()
        if interpretationMethod=="Sklearn Decision Tree":
            tree=SklearnDecisionTree()
        if interpretationMethod=="Sklearn Random Forest":
            tree=SklearnRandomForest()
        if interpretationMethod=="H2o Random Forest":
            tree=H2oRandomForest()
        faulty_attributes=faultyFrame.columns.values[:-2]
        treeModel=tree.train(decisionTreeTrainingFramePreprocessed,faulty_attributes,'label' )
        numberOfTrees=3
        for j in range(numberOfTrees):
            decisionTreeImageUrls.append(tree.visualize(treeModel, faulty_attributes, ['valid','suspicious'],tree_id=j))
        ###############################################
        df = df.style.apply(self.highlight_greaterthan,threshold=faultyThresholdRecords,column=['invalidityScore'], axis=1)
        if invalidityScores[i]>=1.0:
            faulty_records_html.append('<label for="group">Timeseries_'+str(i)+'</label><input type="checkbox" name="Group_faulty" value="'+str(i)+'" checked> </br>'+df.render())
            faulty_cluster_dt_url.append(decisionTreeImageUrls)
            faulty_cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
            faulty_timeseriies_fig_url.append(dataCollection.build_graph(axis,axis,10,90,v_title,h_red,v1_red,faulty_attribute))
        else:
            suspicious_records_html.append('<label for="group">Timeseries_'+str(i)+'</label><input type="checkbox" name="Group_suspicious" value="'+str(i)+'"/> </br>'+df.render())
            suspicious_cluster_dt_url.append(decisionTreeImageUrls)
            suspicious_cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
            suspicious_timeseries_fig_url.append(dataCollection.build_graph(axis,axis,10,90,v_title,h_red,v1_red,faulty_attribute))

    return numberOfClusters,faulty_records_html,suspicious_records_html,faulty_cluster_scores_fig_url,suspicious_cluster_scores_fig_url,faulty_cluster_dt_url,suspicious_cluster_dt_url,faulty_timeseries_fig_url,suspicious_timeseries_fig_url


   def faultInterpretation(self,db,datasetId,constraintDiscoveryMethod,clusteringMethod,interpretationMethod,dataFrame,faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Cluster suspicious records@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #If you want to work with data directly for clustering, use faultyRecordFrame directly. Now it clusters based on invelidity score per feature
    dataFrames=[]
    testing=Testing()
    dataCollection=DataCollection()
    if clusteringMethod=="som":
        #Detect faulty records based on invalidity scores
        faultyInvalidityScoreFrame=testing.detectFaultyRecords(invalidityScoresPerFeature,invalidityScores,faultyThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
        faultyInvalidityScoreFrame.columns=dataFrame.columns.values[:-2]
        som = SOM(5,5, len(faultyInvalidityScoreFrame.columns.values)-1, 400)
        dataFrames=som.clusterFaultyRecords(faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0]],axis=1), faultyRecordFrame)
    elif clusteringMethod=="kprototypes":
        faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame)
        kmeans=H2oKmeans()
        """bestClusteringModel=kmeans.tuneAndTrain(faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1))
        dataFrames=kmeans.clusterFaultyRecords(bestClusteringModel,faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)"""
        bestClusteringModel=kmeans.tuneAndTrain(faultyRecordFramePreprocessed.drop([faultyRecordFramePreprocessed.columns.values[0]],axis=1))
        dataFrames=kmeans.clusterFaultyRecords(bestClusteringModel,faultyRecordFramePreprocessed.drop([faultyRecordFramePreprocessed.columns.values[0]],axis=1), faultyRecordFrame)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Update status of suspicious groups in database@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    db.execute("Update dataRecords_"+datasetId+" set status='invalid' where status like 'actual%' ")    
    i=0
    for dataFrame in dataFrames:
        dataFrame.to_sql('suspicious_i_temp_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious_"+str(i)+ "' where  "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")

        db.execute("Drop table suspicious_i_temp_"+datasetId)       
        i=i+1

    numberOfClusters=i
    faulty_records_html=[]
    suspicious_records_html=[]
    faulty_cluster_scores_fig_url=[]
    suspicious_cluster_scores_fig_url=[]
    faulty_cluster_dt_url=[]
    suspicious_cluster_dt_url=[]
    cluster_interpretation=[]
    treeRules=[]
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@ Add interpretations to groups@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #show the suspicious groups as HTML tables

    for i in range(int(numberOfClusters)):
        faulty_records=dataFrames[i]
        faulty_attributes=dataFrame.columns.values[1:-2]
        cluster_scores=invalidityScoresPerFeature.loc[invalidityScoresPerFeature[dataFrame.columns.values[0]].isin(faulty_records[dataFrame.columns.values[0]])]
        #X=dataFrame.columns.values[1:-2]
        X=dataFrame.columns.values[1:-2]
        Y=cluster_scores.mean().tolist()[1:]
        #indicate the attributes with high invalidity score values
        faulty_attributes_indexes=[i for i,v in enumerate(Y) if v > np.percentile(Y,70)]
        faulty_attributes=X[faulty_attributes_indexes]
        
        #Add decision trees
        normalRecordFrame['label']='valid'
        faulty_records['label']='suspicious'
        decisionTreeTrainingFrame= pd.concat([normalRecordFrame,faulty_records])
        decisionTreeTrainingFramePreprocessed=dataCollection.preprocess(decisionTreeTrainingFrame)
        tree=H2oGradientBoosting()
        if interpretationMethod=="Sklearn Decision Tree":
            tree=SklearnDecisionTree()
        if interpretationMethod=="Sklearn Random Forest":
            tree=SklearnRandomForest()
        if interpretationMethod=="H2o Random Forest":
            tree=H2oRandomForest()
        
        treeModel=tree.train(decisionTreeTrainingFramePreprocessed,faulty_attributes,'label' )
        numberOfTrees=3
        decisionTreeImageUrls=[]
        for j in range(numberOfTrees):
            decisionTreeImageUrls.append(tree.visualize(treeModel, faulty_attributes, ['valid','suspicious'],tree_id=j))
        if faulty_records['invalidityScore'].max()>=1.0:
            #confirmed faulty records
            faulty_records_html.append('<label for="group">Group_'+str(i)+'</label><input type="checkbox" name="Group_faulty" value="'+str(i)+'" checked> </br>'+faulty_records.drop(['status','label'],axis=1).to_html(table_id="group"+str(i)))
            faulty_cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
            faulty_cluster_dt_url.append(decisionTreeImageUrls)
        else:
            suspicious_records_html.append('<label for="group">Group_'+str(i)+'</label><input type="checkbox" name="Group_suspicious" value="'+str(i)+'"/> </br>'+faulty_records.drop(['status','label'],axis=1).to_html(table_id="group"+str(i)))
            suspicious_cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
            suspicious_cluster_dt_url.append(decisionTreeImageUrls)
        #treeCodeLines=tree.treeToCode(treeModel,faulty_attributes)
        #treeRules.append(tree.treeToRules(treeModel,faulty_attributes))
        #cluster_interpretation.append(tree.interpret(treeCodeLines))

    return numberOfClusters,faulty_records_html,suspicious_records_html,faulty_cluster_scores_fig_url,suspicious_cluster_scores_fig_url,faulty_cluster_dt_url,suspicious_cluster_dt_url
