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
        f = request.files['file']
        #f.save(secure_filename(f.filename))

        #dataPath = request.form.get("dataPath")
        interpretationMethod= request.form.getlist("interpretation")
        clusteringMethod= request.form.getlist("clustering")

        error = None

        #if not dataPath:
        #    error = 'Data path is required.'
        if error is None:
            dataCollection=DataCollection()
            dataFrame=dataCollection.importData("datasets/"+f.filename).head(50)
            db=get_db()
            #all the data records are clean by default
            dataFrame['status']='clean'
            datasetId=f.filename.replace('.csv','_').replace("-","_") + str(randint(1,10000))
            dataFrame.to_sql('dataRecords_'+datasetId, con=db, if_exists='replace')           
            return redirect(url_for('DQTestTool.validate', datasetId=datasetId, interpretationMethod=interpretationMethod, clusteringMethod=clusteringMethod))
        flash(error)

    return render_template('import.html')

@bp.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
    return 'file uploaded successfully'

@bp.route('/validate', methods=["GET","POST"])
def validate():
    #Initializations
    datasetId=request.args.get('datasetId')
    db=get_db()
    interpretationMethod=request.args.get('interpretationMethod')
    clusteringMethod=request.args.get('clusteringMethod')
    truePositive=0.0
    truePositiveRate=0.0
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
    #AFdataFrame: stores all actual faults
    AFdataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db)

    dataCollection=DataCollection()
    #remove column id and status for analysis
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0], dataFrame.columns.values[-1]], axis=1))

    #Prepare Training data: Remove actual faults from training data; change their status from clean to actualFaults_i   
    if request.method == "POST":
     numberOfClusters=request.form["numberOfClusters"]
     if numberOfClusters:
        for  i in request.form.getlist('Group'):
            db.execute("Update dataRecords_"+datasetId+" set status='actualFaults_"+str(i)+ "' where status='suspicious_"+str(i)+"'")
    
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId+ " where status='clean' or status like 'suspicious_%'", con=db)
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0], dataFrameTrain.columns.values[-1]], axis=1))
    
    #Tune and Train model
    autoencoder = Autoencoder()
    hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
    l2Opt = [1e-4,1e-2]
    hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
    bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=200,standardize = True,categorical_encoding='auto',export_weights_and_biases=True, quiet_mode=False),dataFrameTrainPreprocessed)

    #Assign invalidity scores per feature
    invalidityScoresPerFeature=autoencoder.assignInvalidityScorePerFeature(bestModel, dataFramePreprocessed)
    
    #Assign invalidity scores per record
    #based on average of attribute's invalidity scores
    #invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)
    #based on max of attribute's invalidity scores
    invalidityScores=invalidityScoresPerFeature.max(axis=1).values.ravel()
    
    #concat record id to the invalidity score per feature
    invalidityScoresPerFeature= pd.concat([dataFrame[dataFrame.columns[0]], invalidityScoresPerFeature], axis=1, sort=False)
    
    #Detect faulty records
    testing=Testing()  
    faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores,np.percentile(invalidityScores,95))#statistics.mean(invalidityScores))#nnp.percentile(invalidityScores,0.5))
    #Detect faulty records based on invalidity scores
    faultyInvalidityScoreFrame=testing.detectFaultyRecords(invalidityScoresPerFeature,invalidityScores,np.percentile(invalidityScores,95))#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
    #Detect normal records
    normalRecordFrame=testing.detectNormalRecords(dataFrame,invalidityScores,np.percentile(invalidityScores,50))#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
    #normalRecordFrame=testing.detectNormalRecordsBasedOnFeatures(dataFrame, invalidityScoresPerFeature, invalidityScores,np.percentile(invalidityScores,0.5))#sum(invalidityScores)/len(invalidityScores))

    #store TPR in database for this run
    if not AFdataFrame.empty:
        truePositiveRate=float(len(AFdataFrame[dataFrame.columns.values[0]].unique().tolist()))/float(faultyRecordFrame.shape[0])
        db.execute('INSERT INTO scores (time, dataset_id, true_positive_rate, false_positive_rate) VALUES (?, ?, ?, ?)',(datetime.datetime.now(), datasetId, truePositiveRate, 1-truePositiveRate))

    #Cluster the faulty records
    #If you want to work with data directly for clustering, use faultyRecordFrame directly. Now it clusters based on invelidity score per feature
    dataFrames=[]
    if clusteringMethod=="som":
        som = SOM(4,4, len(faultyInvalidityScoreFrame.columns.values)-2, 50)
        dataFrames=som.clusterFaultyRecords(faultyInvalidityScoreFrame.drop([faultyInvalidityScoreFrame.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)
    elif clusteringMethod=="kprototypes":
        kmeans=H2oKmeans()
        bestModel=kmeans.tuneAndTrain(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0],'invalidityScore'],axis=1))
        dataFrames=kmeans.clusterFaultyRecords(bestModel,faultyRecordFrame.drop([faultyRecordFrame.columns.values[0],'invalidityScore'],axis=1), faultyRecordFrame)
    #
    
    #Update status of suspicious groups in database
    i=0
    for dataFrame in dataFrames:
        dataFrame.to_sql('suspicious_i_temp_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious_"+str(i)+ "' where "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from suspicious_i_temp_"+datasetId+")")
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
        Y=cluster_scores.mean().tolist()[1:]
        cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))

        #indicate the attributes with high invalidity score values
        faulty_attributes_indexes=[i for i,v in enumerate(Y) if v > np.percentile(Y,90)]
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
        cluster_dt_url.append(tree.visualize(treeModel,faulty_attributes,['valid','suspicious']))
        treeCodeLines=tree.treeToCode(treeModel,faulty_attributes)
        treeRules.append(tree.treeToRules(treeModel,faulty_attributes))
        cluster_interpretation.append(tree.interpret(treeCodeLines))
       
    if request.method == 'POST':
        knownFaults=""
        if request.files:
            f = request.files['file']
            knownFaults='datasets/PD/'+f.filename
            f.save(knownFaults)
        #knownFaults = request.form.get("knownFaults")
        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId, knownFaults=knownFaults))
         
    return render_template('validate.html', data='@'.join(faulty_records_html),  numberOfClusters=numberOfClusters, fig_urls=cluster_scores_fig_url,cluster_dt_url=cluster_dt_url, cluster_interpretation=cluster_interpretation, treeRules=treeRules)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    knownFaults=request.args.get('knownFaults')

    #A
    faulty_records=pd.read_sql(sql="SELECT distinct * FROM dataRecords_"+datasetId+" where status like 'suspicious_%' or status like 'actualFaults_%'", con=db)
    A=set(faulty_records[faulty_records.columns.values[0]].astype(str).tolist())
    
    #AF
    AFdataFrame=pd.read_sql(sql="select distinct * from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db )  
    AF=set(AFdataFrame[AFdataFrame.columns.values[0]].astype(str).unique().tolist())
    
    #E
    E=set()
    if knownFaults:
        dataCollection=DataCollection()
        E=dataCollection.csvToSet(str(knownFaults))
    

    evaluation=Evaluation()
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    
    
    #statistics    
    #TPR=evaluation.truePositiveRate(A,TF)
    TPR=0
    #TPGR=evaluation.truePositiveGrowthRate(score)
    TPGR=0
    #NR=evaluation.numberOfRuns(score)
    NR=0
    PD=evaluation.previouslyDetectedFaultyRecords(A,E)
    ND=evaluation.newlyDetectedFaultyRecords(A, E, AF)
    UD=evaluation.unDetectedFaultyRecords(A, E)

    #TODO:save

    return render_template('evaluation.html',  score=score.to_html(), TF=float(len(AF)), A=float(len(A)), TPR=TPR, NR=NR, TPGR=TPGR, E=float(len(E)), PD=PD, ND=ND, UD=UD)

