import os
from random import *
import functools
from backendClasses.DataCollection import DataCollection
from backendClasses.PatternDiscovery import PatternDiscovery
from backendClasses.SklearnDecisionTree import SklearnDecisionTree
from backendClasses.SklearnRandomForest import SklearnRandomForest
from backendClasses.H2oGradientBoosting import H2oGradientBoosting
from backendClasses.H2oRandomForest import H2oRandomForest
from backendClasses.H2oKmeans import H2oKmeans
from backendClasses.SOM import SOM
from backendClasses.Testing import Testing
import h2o
import numpy as np
from backendClasses.Autoencoder import Autoencoder
from backendClasses.Pyod import Pyod
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
from DataQualityTestTool.db import get_db
from backendClasses.Evaluation import Evaluation
import datetime

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import statistics
import pyod
from keras.models import load_model

bp = Blueprint('DQTestTool', __name__, url_prefix='/DQTestTool')


@bp.route('/import', methods=["GET","POST"])
def importDataFrame():
    """
    Import CSV data 
    """
    if request.method == 'POST':
        trainedModelFilePath=""
        knownFaultsFilePath=""
        constraintDiscoveryMethod=request.form.get("constraintDiscovery")
        interpretationMethod= request.form.get("interpretation")
        clusteringMethod= request.form.get("clustering")
        error = None
        if request.files.get('trainedModel', None):
            trainedModelFile=request.files['trainedModel']
            trainedModelFilePath='static/model/'+trainedModelFile.filename
            trainedModelFile.save(trainedModelFilePath)
        if request.files.get('knownFaults', None):
            knownFaultsFile=request.files['knownFaults']
            knownFaultsFilePath="datasets/PD/"+knownFaultsFile.filename
            knownFaultsFile.save(knownFaultsFilePath)
        if request.files.get('dataRecords', None):
            dataRecordsFile = request.files['dataRecords']
            dataRecordsFilePath="datasets/"+dataRecordsFile.filename
            dataRecordsFile.save(dataRecordsFilePath)
        else:
            error="No data file is selected"
        if error is None:
            dataCollection=DataCollection()
            dataFrame=dataCollection.importData(dataRecordsFilePath)
            db=get_db()
            #all the data records are clean by default
            dataFrame['status']='clean'
            dataFrame['invalidityScore']=0.0
            datasetId=dataRecordsFile.filename.replace('.csv','_').replace("-","_").replace(" ","_" ) + str(randint(1,10000))
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

            return redirect(url_for('DQTestTool.validate', datasetId=datasetId, constraintDiscoveryMethod=constraintDiscoveryMethod, interpretationMethod=interpretationMethod, clusteringMethod=clusteringMethod))
        flash(error)

    return render_template('import.html')


@bp.route('/validate', methods=["GET","POST"])
def validate():
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@Initializations@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    datasetId=request.args.get('datasetId')
    db=get_db()
    constraintDiscoveryMethod=request.args.get('constraintDiscoveryMethod')
    interpretationMethod=request.args.get('interpretationMethod')
    clusteringMethod=request.args.get('clusteringMethod')
    truePositive=0.0
    truePositiveRate=0.0
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    NRDataFrame=pd.read_sql(sql="SELECT count(*) FROM scores where dataset_id like '"+datasetId+"'", con=db)
    NR=NRDataFrame[NRDataFrame.columns.values[0]].values[0]
    trainedModelFilePathDataFrame=pd.read_sql(sql="select trainedModelFilePath from hyperParameters_"+datasetId, con=db)
    trainedModelFilePath=trainedModelFilePathDataFrame[trainedModelFilePathDataFrame.columns.values[0]].values[0]

    dataCollection=DataCollection()
    #dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'status','invalidityScore'], axis=1))
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'invalidityScore'], axis=1))

    numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious%'",con=db)
    numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
    suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious%'", con=db)
    AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])
    if request.method == "POST":
        #select actual faluts from previous run before updating the database - we need this information to measure the false negative rate
        AFdataFrameOld=pd.read_sql(sql="select distinct "+dataFrame.columns.values[0]+" from actualFaults_"+datasetId, con=db)

        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId))
        
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #@@@@@@@@@@@@@@Incorporate domain knowledge@@@@@@@@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        numberOfClusters=request.form["numberOfClusters"]
        maxInvalidityScoreOfNormalData=[]
        if numberOfClusters:
            for i in range(int(numberOfClusters)):
                if str(i) in request.form.getlist('Group'):
                    db.execute("Update dataRecords_"+datasetId+" set  status='actualFaults_"+str(i)+ "' where status='suspicious_"+str(i)+"'")
                    
                else:
                    db.execute("Update dataRecords_"+datasetId+" set  status='valid' where status='suspicious_"+str(i)+"'")
                    #db.execute("Update dataRecords_"+datasetId+" set  status='clean' where status='suspicious_"+str(i)+"'")

        
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
    AFdataFrame=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db)

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
        hiddenOpt=[[50,50]]
        l2Opt = [1e-2]
        hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
        MLmodels[constraintDiscoveryMethod]=H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=60,standardize = True,categorical_encoding='auto',export_weights_and_biases=True, quiet_mode=False,hidden=[50,50], l2=1e-4, train_samples_per_iteration=-1,pretrained_autoencoder=preTrainedModel, rate=0.1)
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
        print invalidityScores
        

    invalidityScoresWithId= pd.concat([dataFrame[dataFrame.columns[0]], pd.DataFrame(invalidityScores, columns=['invalidityScore'])], axis=1, sort=False)
    for index, row in invalidityScoresWithId.iterrows():
        db.execute("Update dataRecords_"+datasetId+" set invalidityScore="+str(row['invalidityScore'])+" where "+dataFrame.columns.values[0]+"="+str(row[dataFrame.columns.values[0]]))
    
    invalidityScoresPerFeature= pd.concat([dataFrame[dataFrame.columns[0]], invalidityScoresPerFeature], axis=1, sort=False)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@Identify Threshold@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    numberOfKnownFaultsDataFrame=pd.read_sql(sql="SELECT count(*) FROM knownFaults_"+datasetId, con=db)
    numberOfKnownFaults=numberOfKnownFaultsDataFrame[numberOfKnownFaultsDataFrame.columns.values[0]].values[0]
    faultyThreshold=np.percentile(invalidityScores,90)        
    if numberOfKnownFaults>0:
        faultyThreshold=np.percentile(invalidityScores, 85-(100*(float(numberOfKnownFaults)/float(len(dataFrame)))))
    
    aDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    a=(aDataFrame[aDataFrame.columns.values[0]].values[0])
    bDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'actualFault%'",con=db)
    b=(bDataFrame[bDataFrame.columns.values[0]].values[0])
    cDataFrame=pd.read_sql(sql="select min(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    c=(cDataFrame[cDataFrame.columns.values[0]].values[0])
    dDataFrame=pd.read_sql(sql="select max(invalidityScore) from dataRecords_"+datasetId+ " where status like 'valid' or status like 'clean'",con=db)
    d=(dDataFrame[dDataFrame.columns.values[0]].values[0])
    
    #it is not ( either the first run or scores of valids are greater than scores of invalids)
    if b!=0 and  b>d:   
        #if valid/invalids are well separated and the number of detected faults are not too small
        """if a>=d and d< np.percentile(invalidityScores,95):
            faultyThreshold=d"""
        if d>a and d<b:
            faultyThreshold=max(a,faultyThreshold)
        elif a>=d:
            faultyThreshold=min(a,np.percentile(invalidityScores, 98-(100*(float(numberOfKnownFaults)/float(len(dataFrame))))))

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
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Cluster suspicious records@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #If you want to work with data directly for clustering, use faultyRecordFrame directly. Now it clusters based on invelidity score per feature
    dataFrames=[]
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
            X=dataFrame.columns.values[1:-1]
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
    return render_template('validate.html', data='@'.join(faulty_records_html), datasetId=datasetId, numberOfClusters=numberOfClusters, fig_urls=cluster_scores_fig_url,cluster_dt_url=cluster_dt_url, cluster_interpretation=cluster_interpretation, treeRules=treeRules, bestModelFile='/static/model/'+bestModelFileName)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    evaluation=Evaluation()
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    
    #A
    suspiciousRecords=pd.read_sql(sql="SELECT distinct * FROM dataRecords_"+datasetId+" where status like 'suspicious_%' or status like 'actualFaults_%'", con=db)
    A=set(suspiciousRecords[suspiciousRecords.columns.values[0]].astype(str).tolist())
    #AF
    actualFaults=pd.read_sql(sql="select distinct * from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db )  
    AF=set(actualFaults[actualFaults.columns.values[0]].astype(str).unique().tolist())
    #E
    knownFaults=pd.read_sql(sql="select distinct * from knownFaults_"+datasetId,con=db)
    E=set(knownFaults[knownFaults.columns.values[0]].astype(str).tolist())
    
    TPGR=evaluation.truePositiveGrowthRate(score)
    return render_template('evaluation.html',  score=score.to_html(),TPGR=TPGR,A=len(A), AF=len(AF), E=len(E))

