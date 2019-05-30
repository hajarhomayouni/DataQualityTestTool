import os
from random import *
import functools
from backendClasses.DataCollection import DataCollection
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

bp = Blueprint('DQTestTool', __name__, url_prefix='/DQTestTool')


@bp.route('/import', methods=["GET","POST"])
def importDataFrame():
    """
    Import CSV data 
    """
    if request.method == 'POST':
        trainedModelFilePath=""
        interpretationMethod= request.form.getlist("interpretation")
        clusteringMethod= request.form.getlist("clustering")
        error = None
        if request.files.get('trainedModel', None):
            trainedModelFile=request.files['trainedModel']
            trainedModelFilePath='static/model/'+trainedModelFile.filename
            trainedModelFile.save(trainedModelFilePath)
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
            dataFrame['invalidityScore']=0.0
            dataFrame['status']='clean'
            datasetId=dataRecordsFile.filename.replace('.csv','_').replace("-","_").replace(" ","_" ) + str(randint(1,10000))
            dataFrame.to_sql('dataRecords_'+datasetId, con=db, if_exists='replace')           
            return redirect(url_for('DQTestTool.validate',trainedModelFilePath=trainedModelFilePath, datasetId=datasetId, interpretationMethod=interpretationMethod, clusteringMethod=clusteringMethod))
        flash(error)

    return render_template('import.html')


@bp.route('/validate', methods=["GET","POST"])
def validate():
    #Initializations
    datasetId=request.args.get('datasetId')
    trainedModelFilePath=request.args.get('trainedModelFilePath')
    db=get_db()
    interpretationMethod=request.args.get('interpretationMethod')
    clusteringMethod=request.args.get('clusteringMethod')
    truePositive=0.0
    truePositiveRate=0.0
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    #select actual faluts from previous run before updating the database - we need this information to measure the false negative rate
    AFdataFrameOld=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db)

    dataCollection=DataCollection()
    #remove column id for analysis
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1))
    faultyThresholdByExpert=0.0
    if request.method == "POST":
        knownFaults=""
        if request.files:
            f = request.files['file']
            knownFaults='datasets/PD/'+f.filename
            f.save(knownFaults)
        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId, knownFaults=knownFaults))
        #Prepare Training data: Remove actual faults from training data; change their status from clean to actualFaults_i   
        trainedModelFilePath=""     
        numberOfClusters=request.form["numberOfClusters"]
        maxInvalidityScoreOfNormalData=[]
        if numberOfClusters:
            for i in range(int(numberOfClusters)):
                if str(i) in request.form.getlist('Group'):
                    db.execute("Update dataRecords_"+datasetId+" set status='actualFaults_"+str(i)+ "' where status='suspicious_"+str(i)+"'")
                else:
                    maxInvalidityScoreOfNormalData.append(pd.read_sql(sql="select invalidityScore from dataRecords_"+datasetId+" where status='suspicious_"+str(i)+"' or status='actualFaults_"+str(i)+"'",con=db).iloc[0,0])
                    db.execute("Update dataRecords_"+datasetId+" set status='clean' where status='suspicious_"+str(i)+"' or status='actualFaults_"+str(i)+"'")
            if len(maxInvalidityScoreOfNormalData)>0:
                faultyThresholdByExpert=max(maxInvalidityScoreOfNormalData)


    #select actual faluts from the current run after updating the database - we need this information to measure the false negative rate
    AFdataFrame=pd.read_sql(sql="select "+dataFrame.columns.values[0]+" from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db)
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where status='clean' or status like 'suspicious_%'", con=db)
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0]], axis=1))
    #Tune and Train model
    h2o.init()   
    autoencoder = Autoencoder()
    bestConstraintDiscoveryModel=H2OAutoEncoderEstimator()
    if trainedModelFilePath!="":
        bestConstraintDiscoveryModel = h2o.load_model(trainedModelFilePath)
    else:
        hiddenOpt = [[5],[50],[100],[5,5],[50,50],[100,100], [5,5,5],[50,50,50],[100,100,100]]
        l2Opt = [1e-4,1e-2]
        hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
        bestConstraintDiscoveryModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=200,standardize = True,categorical_encoding='auto',export_weights_and_biases=True, quiet_mode=False),dataFrameTrainPreprocessed)

    #Assign invalidity scores per feature
    print "******dataFramePreprocessed"
    print dataFramePreprocessed
    invalidityScoresPerFeature=autoencoder.assignInvalidityScorePerFeature(bestConstraintDiscoveryModel, dataFramePreprocessed)
    print "****invalidityScoresPerFeature***" 
    print invalidityScoresPerFeature
    #Assign invalidity scores per record
    #based on average of attribute's invalidity scores
    #invalidityScores=autoencoder.assignInvalidityScore(bestConstraintDiscoveryModel, dataFramePreprocessed)
    #based on max of attribute's invalidity scores
    invalidityScores=invalidityScoresPerFeature.max(axis=1).values.ravel()
    print "***invalidityScores**************"
    print invalidityScores

    #concat record id to the invalidity score
    invalidityScoresWithId= pd.concat([dataFrame[dataFrame.columns[0]], pd.DataFrame(invalidityScores, columns=['invalidityScore'])], axis=1, sort=False)
    print "*****invalidityScoresWithId******"
    print invalidityScoresWithId
    for index, row in invalidityScoresWithId.iterrows():
        db.execute("Update dataRecords_"+datasetId+" set invalidityScore="+str(row['invalidityScore'])+" where "+dataFrame.columns.values[0]+"="+str(row[dataFrame.columns.values[0]]))
    # 
    
    #concat record id to the invalidity score per feature
    invalidityScoresPerFeature= pd.concat([dataFrame[dataFrame.columns[0]], invalidityScoresPerFeature], axis=1, sort=False)
    print "******invalidityScoresPerFeature after concatinating with id"
    print invalidityScoresPerFeature

    faultyThreshold=np.percentile(invalidityScores,50)    
    faultyThreshold=max(faultyThreshold, faultyThresholdByExpert)
    normalThreshold=np.percentile(invalidityScores,50)
    #Detect faulty records
    testing=Testing()  
    faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores,faultyThreshold)#statistics.mean(invalidityScores))#nnp.percentile(invalidityScores,0.5))
    print "*********faultyRecordFrame"
    print faultyRecordFrame
    #Detect faulty records based on invalidity scores
    faultyInvalidityScoreFrame=testing.detectFaultyRecords(invalidityScoresPerFeature,invalidityScores,faultyThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
    print "**********faultyInvalidityScoreFrame"
    print faultyInvalidityScoreFrame
    columnNames=faultyRecordFrame.columns
    #columnNames.remove('status')
    faultyInvalidityScoreFrame.columns=columnNames#.drop('status','invalidityScore')
    #Detect normal records
    normalRecordFrame=testing.detectNormalRecords(dataFrame,invalidityScores,normalThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
    #normalRecordFrame=testing.detectNormalRecordsBasedOnFeatures(dataFrame, invalidityScoresPerFeature, invalidityScores,np.percentile(invalidityScores,0.5))#sum(invalidityScores)/len(invalidityScores))

    #store different scores in database for this run
    truePositiveRate=0.0
    if not AFdataFrame.empty:
        truePositiveRate=float(len(AFdataFrame[dataFrame.columns.values[0]].unique().tolist()))/float(faultyRecordFrame.shape[0])
    falsePositiveRate=1.0-truePositiveRate
    falseNegativeRate=0.0
    diffDetection=len(set(AFdataFrameOld[dataFrame.columns.values[0]].unique().tolist()).difference(set(AFdataFrame[dataFrame.columns.values[0]].unique().tolist()))) 
    oldDetected=len(set(AFdataFrameOld[dataFrame.columns.values[0]].unique().tolist())) 
    if not AFdataFrameOld.empty:
        falseNegativeRate=float(diffDetection)/float(oldDetected)
    trueNegativeRate=1-falseNegativeRate
    db.execute('INSERT INTO scores (time, dataset_id, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate) VALUES (?, ?, ?, ?, ?,?)',(datetime.datetime.now(), datasetId, truePositiveRate, falsePositiveRate,trueNegativeRate, falseNegativeRate))

    #Cluster the faulty records
    #If you want to work with data directly for clustering, use faultyRecordFrame directly. Now it clusters based on invelidity score per feature
    dataFrames=[]
    if clusteringMethod=="som":
        som = SOM(4,4, len(faultyInvalidityScoreFrame.columns.values)-2, 400)
        dataFrames=som.clusterFaultyRecords(faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)
    elif clusteringMethod=="kprototypes":
        faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame)
        kmeans=H2oKmeans()
        """bestClusteringModel=kmeans.tuneAndTrain(faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1))
        dataFrames=kmeans.clusterFaultyRecords(bestClusteringModel,faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)"""
        bestClusteringModel=kmeans.tuneAndTrain(faultyRecordFramePreprocessed.drop([faultyRecordFramePreprocessed.columns.values[0],'invalidityScore'],axis=1))
        dataFrames=kmeans.clusterFaultyRecords(bestClusteringModel,faultyRecordFramePreprocessed.drop([faultyRecordFramePreprocessed.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)

    #Update status of suspicious groups in database
    i=0
    for dataFrame in dataFrames:
        dataFrame.to_sql('suspicious_i_temp_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious_"+str(i)+ "' where status='clean' and "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")

        db.execute("Update dataRecords_"+datasetId+" set status='actualFaults_"+str(i)+ "' where status='actualtFaults_%' and "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")
        db.execute("Drop table suspicious_i_temp_"+datasetId)       
        i=i+1
    numberOfClusters=i
    faulty_records_html=[]
    cluster_scores_fig_url=[]
    cluster_dt_url=[]
    cluster_interpretation=[]
    treeRules=[]
    
    #show the suspicious groups as HTML tables
    for i in range(int(numberOfClusters)):
        faulty_records=dataFrames[i]
        faulty_records_html.append(faulty_records.to_html())
        
        #Show descriptive graph for each group
        cluster_scores=invalidityScoresPerFeature.loc[invalidityScoresPerFeature[dataFrame.columns.values[0]].isin(faulty_records[dataFrame.columns.values[0]])]
        #exclude status and invalidityScore columns
        X=dataFrame.columns.values[1:-2]
        Y=cluster_scores.mean().tolist()[1:-2]
        cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))

        #indicate the attributes with high invalidity score values
        faulty_attributes_indexes=[i for i,v in enumerate(Y) if v > np.percentile(Y,70)]
        faulty_attributes=X[faulty_attributes_indexes]
        
        #Interpret each cluster
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
        #number of generated trees = 3
        numberOfTrees=3
        decisionTreeImageUrls=[]
        for i in range(numberOfTrees):
            decisionTreeImageUrls.append(tree.visualize(treeModel, faulty_attributes, ['valid','suspicious'],tree_id=i))
        #
        cluster_dt_url.append(decisionTreeImageUrls)
        treeCodeLines=tree.treeToCode(treeModel,faulty_attributes)
        treeRules.append(tree.treeToRules(treeModel,faulty_attributes))
        cluster_interpretation.append(tree.interpret(treeCodeLines))
       
    bestModelPath = h2o.save_model(model=bestConstraintDiscoveryModel, path="static/model/", force=True)         
    bestModelFileName=bestModelPath.split('/')[len(bestModelPath.split('/'))-1]
    return render_template('validate.html', data='@'.join(faulty_records_html), datasetId=datasetId, numberOfClusters=numberOfClusters, fig_urls=cluster_scores_fig_url,cluster_dt_url=cluster_dt_url, cluster_interpretation=cluster_interpretation, treeRules=treeRules, bestModelFile='/static/model/'+bestModelFileName)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    knownFaults=request.args.get('knownFaults')

    #A
    suspiciousRecords=pd.read_sql(sql="SELECT distinct * FROM dataRecords_"+datasetId+" where status like 'suspicious_%' or status like 'actualFaults_%'", con=db)
    A=set(suspiciousRecords[suspiciousRecords.columns.values[0]].astype(str).tolist())
    print "@@A@@@@@@@@@@@@@"
    print A
    
    #AF
    actualFaults=pd.read_sql(sql="select distinct * from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db )  
    AF=set(actualFaults[actualFaults.columns.values[0]].astype(str).unique().tolist())
    
    #E
    E=set()
    if knownFaults:
        dataCollection=DataCollection()
        E=dataCollection.csvToSet(str(knownFaults))
    print "@@@@E@@@@@@@@"
    print E

    evaluation=Evaluation()
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    
    
    #statistics    
    TPR=evaluation.truePositiveRate(A,AF)
    TPGR=evaluation.truePositiveGrowthRate(score)
    NR=evaluation.numberOfRuns(score)
    PD=evaluation.previouslyDetectedFaultyRecords(A,E)
    SD=evaluation.newlyDetectedFaultyRecords(A, E, A)
    ND=evaluation.newlyDetectedFaultyRecords(A, E, AF)
    UD=evaluation.unDetectedFaultyRecords(A, E)

    db.execute("Update scores set previously_detected="+str(PD)+", suspicious_detected="+str(SD)+ ", undetected="+str(UD)+", newly_detected="+str(ND)+" where dataset_id like '"+datasetId+"'")
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    

    return render_template('evaluation.html',  score=score.to_html(), TF=float(len(AF)), A=float(len(A)), TPR=TPR, NR=NR, TPGR=TPGR, E=float(len(E)), PD=PD, ND=ND, UD=UD)

