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
from .Pyod import Pyod
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
from .Evaluation import Evaluation
import datetime


import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import statistics
import pyod
from keras.models import load_model
from tsfresh import select_features,extract_features
from tsfresh.utilities.dataframe_functions import impute

class DQTestToolHelper:
    
   def importData(self,db,dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath):
    dataCollection=DataCollection()
    dataFrame=dataCollection.importData(dataRecordsFilePath)#.head(20)
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


   def constraintDiscoveryAndFaultDetection(self,db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,truePositiveRateGroup):
    truePositive=0.0
    truePositiveRate=0.0
    NRDataFrame=pd.read_sql(sql="SELECT count(*) FROM scores where dataset_id like '"+datasetId+"'", con=db)
    NR=NRDataFrame[NRDataFrame.columns.values[0]].values[0]
    trainedModelFilePathDataFrame=pd.read_sql(sql="select trainedModelFilePath from hyperParameters_"+datasetId, con=db)
    trainedModelFilePath=trainedModelFilePathDataFrame[trainedModelFilePathDataFrame.columns.values[0]].values[0]

    dataCollection=DataCollection()
    dataFramePreprocessed=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop(['invalidityScore','status'], axis=1))
    else:
        dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'invalidityScore','status'], axis=1))

    #suspicious is zero because we are deciding based on the expert idea in previous time not based on our idea
    #y=dataFrame['status'].replace('valid',-1).replace('invalid',1).replace('actual*',1,regex=True).replace('clean',0)
    #dataFramePreprocessed['status']=y

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #prepare training data@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #select actual faluts from the current run after updating the database - we need this information to measure the false negative rate
    AFdataFrame=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults%'", con=db)

    AFdataFrame.to_sql('actualFaults_'+datasetId, con=db, if_exists='append')
    #dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where status not like 'actual%' and status not like 'invalid'", con=db)
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    
    """validDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'valid'", con=db)
    for i in range(50):
        dataFrameTrain=dataFrameTrain.append(validDataFrame, ignore_index=True)"""
    dataFrameTrainPreprocessed=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop(['invalidityScore','status'], axis=1))
    else:
        dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0],'invalidityScore','status'], axis=1))
    y=dataFrameTrain['status'].replace('valid',-1).replace('invalid',1).replace('actual*',1,regex=True).replace('clean',0)
    #dataFrameTrainPreprocessed['status']=y
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@Set up constraint model parameters@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    patternDiscovery = PatternDiscovery()
    bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain("","","")
    hyperParameters=[]
    n=len(dataFrameTrainPreprocessed.columns.values)
    hidden_neurons=[ n/2, n/2]
    MLmodels={"LSTMAutoencoder":"","H2O_Autoencoder":H2OAutoEncoderEstimator(), "KNN": pyod.models.knn.KNN(), "SO_GAAL":pyod.models.so_gaal.SO_GAAL(),"MO_GAAL": pyod.models.mo_gaal.MO_GAAL(), "PCA": pyod.models.pca.PCA(), "MCD": pyod.models.mcd.MCD(),"OCSVM": pyod.models.ocsvm.OCSVM(), "Pyod_Autoencoder":pyod.models.auto_encoder.AutoEncoder(hidden_neurons=hidden_neurons, epochs=10,contamination=0.5,l2_regularizer=0.5, batch_size=len(dataFrameTrain)), "LOF":pyod.models.lof.LOF()}
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
    elif constraintDiscoveryMethod=="LSTMAutoencoder":
        patternDiscovery=LSTMAutoencoder()
    else:
        patternDiscovery=Pyod()

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@Train Model@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    dataFrameTimeseries=pd.DataFrame()
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        bestConstraintDiscoveryModel,dataFrameTimeseries=patternDiscovery.tuneAndTrain(dataFrameTrainPreprocessed)
    else:
        bestConstraintDiscoveryModel=patternDiscovery.tuneAndTrain(hyperParameters,MLmodels[constraintDiscoveryMethod],dataFrameTrainPreprocessed)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@save the trained model for next run use@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    bestModelFileName=""
    if "H2O_Autoencoder" in constraintDiscoveryMethod:
        bestModelPath = h2o.save_model(model=bestConstraintDiscoveryModel, path="static/model/", force=True)         
        bestModelFileName=bestModelPath.split('/')[len(bestModelPath.split('/'))-1]
        trainedModelFilePath='/static/model/'+bestModelFileName
        db.execute("update hyperparameters_"+str(datasetId)+" set trainedModelFilePath='/home/hajar/DataQualityTestTool"+trainedModelFilePath+"'")
    elif "LSTMAutoencoder" in constraintDiscoveryMethod:
        print ("TODO")
    else:
        bestConstraintDiscoveryModel.model_.save("static/model/pyod_"+str(datasetId)+".h5")
        db.execute("update hyperparameters_"+str(datasetId)+" set trainedModelFilePath='static/model/pyod_"+str(datasetId)+"'")

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Assign invalidity scores@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    invalidityScores=[]
    invalidityScoresPerFeature=[]
    yhatWithInvalidityScores=[]
    XWithInvalidityScores=[]
    mse_attributes=[]
    if "LSTMAutoencoder" in constraintDiscoveryMethod:
        mse_timeseries, mse_records, mse_attributes,yhatWithInvalidityScores,XWithInvalidityScores=patternDiscovery.assignInvalidityScore(bestConstraintDiscoveryModel,dataFramePreprocessed.to_numpy(),y)
        invalidityScores=mse_timeseries
    else:
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
        if constraintDiscoveryMethod=="H2O_Autoencoder":
            faultyThreshold=np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))
        elif constraintDiscoveryMethod=="LSTMAutoencoder":
            faultyThreshold=np.percentile(invalidityScores, (100-(100*(float(numberOfKnownFaults*3)/float(len(dataFrameTimeseries))))))
    
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
                    faultyThreshold=max(0,min(a,np.percentile(invalidityScores, 100-(100*(float(numberOfKnownFaults*3)/float(len(dataFrameTimeseries)))))))
                if constraintDiscoveryMethod=="H2OAutoencoder":
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
        normalTimeseriesIndexes=np.where(invalidityScores<faultyThreshold)
        faultyRecordsInTimeseries=pd.DataFrame()
        normalRecordsInTimeseries=pd.DataFrame()
        for i in faultyTimeseriesIndexes[0]:
            faultyRecordsInTimeseries=pd.concat([faultyRecordsInTimeseries, pd.DataFrame(XWithInvalidityScores[i][:,0])])
        for j in normalTimeseriesIndexes[0]:
            normalRecordsInTimeseries=pd.concat([normalRecordsInTimeseries, pd.DataFrame(XWithInvalidityScores[i][:,0])])
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
    db.execute('INSERT INTO scores (time, dataset_id,previously_detected,suspicious_detected,undetected,newly_detected, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate, true_positive_rate_timeseries) VALUES (?,?,?,?,?, ?, ?, ?, ?,?,?)',(datetime.datetime.now(), datasetId,PD,SD,UD,ND,truePositiveRate, falsePositiveRate,trueNegativeRate, falseNegativeRate, truePositiveRateGroup))
    return faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y


   def faultyTimeseriesInterpretation(self,db,interpretationMethod,datasetId,dataFramePreprocessed,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFrameTimeseries,y):
    dataCollection=DataCollection() 
    numberOfClusters=len(faultyTimeseriesIndexes[0])
    faulty_records_html=[]
    cluster_scores_fig_url=[]
    db.execute("Update dataRecords_"+datasetId+" set status='invalid' where status like 'actual%' ")    
    print("dataFrameTimeseries")
    print(dataFrameTimeseries)
    #timeseriesFeatures=extract_features(dataFrameTimeseries.drop([dataFrameTimeseries.columns.values[0]],axis=1),column_id="timeseriesId",column_sort="time")
    timeseriesFeatures=extract_features(dataFrameTimeseries,column_id="timeseriesId",column_sort="time")
    print("timeseries features***************")
    print(timeseriesFeatures)
    normalTimeseries=pd.DataFrame(data=np.transpose(normalTimeseriesIndexes),columns=[dataFrameTimeseries.columns.values[0]])
    normalTimeseries["label"]=0
    normalFrame=pd.merge(timeseriesFeatures,normalTimeseries, on=[dataFrameTimeseries.columns.values[0]])
    print("normalFrame")
    print(normalFrame)
    #
    """normalTimeseries=pd.DataFrame(data=np.transpose(normalTimeseriesIndexes),columns=[dataFrameTimeseries.columns.values[0]])
    normalTimeseries["label"]=0
    normalDataFrameTimeseries=pd.merge(dataFrameTimeseries,normalTimeseries, on=[dataFrameTimeseries.columns.values[0]])
    faultyTimeseries=pd.DataFrame(data=np.transpose(faultyTimeseriesIndexes),columns=[dataFrameTimeseries.columns.values[0]])
    faultyTimeseries["label"]=1
    faultyDataFrameTimeseries=pd.merge(dataFrameTimeseries,faultyTimeseries, on=[dataFrameTimeseries.columns.values[0]])
    dataFrameTimeseries=pd.concat([normalDataFrameTimeseries,faultyDataFrameTimeseries])
    print("dataFrameTimeseries")
    print(dataFrameTimeseries)"""
    #

    cluster_dt_url=[]
    index=0
    for i in faultyTimeseriesIndexes[0]:
        df = pd.DataFrame(XWithInvalidityScores[i], columns=np.append(dataFramePreprocessed.columns.values,'invalidityScore'))
        faulty_records_html.append(df.to_html())
        X=dataFramePreprocessed.columns.values[2:]
        Y=mse_attributes[i][2:]
        cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #Update status of suspicious groups in database@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        df.to_sql('suspicious_i_temp_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious_"+str(index)+ "' where  "+dataFramePreprocessed.columns.values[0]+" in (select "+dataFramePreprocessed.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")
        db.execute("Drop table suspicious_i_temp_"+datasetId)
        index=index+1
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #Add Decision Tree for each Timesereis
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        faulty_attributes=timeseriesFeatures.columns.values
        #print("faulty_attributes********")
        #print(faulty_attributes)
        faultyTimeseries=pd.DataFrame(data=[i],columns=["id"])
        faultyTimeseries["label"]=1
        faultyFrame=pd.merge(timeseriesFeatures,faultyTimeseries, on='id')
        decisionTreeTrainingFrame=pd.concat([normalFrame,faultyFrame])
        print("decision Tree Training Frame***************")
        print (decisionTreeTrainingFrame)
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
        for i in range(numberOfTrees):
            decisionTreeImageUrls.append(tree.visualize(treeModel, faulty_attributes, ['valid','suspicious'],tree_id=i))
        cluster_dt_url.append(decisionTreeImageUrls)
        """treeCodeLines=tree.treeToCode(treeModel,faulty_attributes)
        treeRules.append(tree.treeToRules(treeModel,faulty_attributes))
        cluster_interpretation.append(tree.interpret(treeCodeLines))"""


    return numberOfClusters,faulty_records_html,cluster_scores_fig_url,cluster_dt_url,"",""


   def faultInterpretation(self,db,datasetId,constraintDiscoveryMethod,clusteringMethod,interpretationMethod,dataFrame,faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName):
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
    cluster_scores_fig_url=[]
    cluster_dt_url=[]
    cluster_interpretation=[]
    treeRules=[]
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@ Add interpretations to groups@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #show the suspicious groups as HTML tables
    for i in range(int(numberOfClusters)):
        faulty_records=dataFrames[i]
        faulty_records_html.append(faulty_records.drop(['status'],axis=1).to_html())
        faulty_attributes=dataFrame.columns.values[1:-2]
        if constraintDiscoveryMethod=="H2O_Autoencoder":
            cluster_scores=invalidityScoresPerFeature.loc[invalidityScoresPerFeature[dataFrame.columns.values[0]].isin(faulty_records[dataFrame.columns.values[0]])]
            #X=dataFrame.columns.values[1:-2]
            X=dataFrame.columns.values[1:-2]
            Y=cluster_scores.mean().tolist()[1:]
            cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
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
        for i in range(numberOfTrees):
            decisionTreeImageUrls.append(tree.visualize(treeModel, faulty_attributes, ['valid','suspicious'],tree_id=i))
        cluster_dt_url.append(decisionTreeImageUrls)
        treeCodeLines=tree.treeToCode(treeModel,faulty_attributes)
        treeRules.append(tree.treeToRules(treeModel,faulty_attributes))
        cluster_interpretation.append(tree.interpret(treeCodeLines))

    return numberOfClusters,faulty_records_html,cluster_scores_fig_url,cluster_dt_url,cluster_interpretation,treeRules
