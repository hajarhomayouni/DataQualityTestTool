from random import *
import functools
from backendClasses.DataCollection import DataCollection
from backendClasses.DecisionTree import DecisionTree
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

bp = Blueprint('DQTestTool', __name__, url_prefix='/DQTestTool')


@bp.route('/import', methods=["GET","POST"])
def importDataFrame():
    """
    Import CSV data 
    """
    if request.method == 'POST':
        dataPath = request.form.get("dataPath")
        error = None

        if not dataPath:
            error = 'Data path is required.'

        if error is None:
            dataCollection=DataCollection()
            dataFrame=dataCollection.importData("datasets/"+dataPath)
            db=get_db()
            dataFrame.to_sql('dataRecords', con=db, if_exists='replace')
            dataFrame.to_sql('trainingRecords', con=db, if_exists='replace')
            datasetId=dataPath.replace('.csv','_') + str(randint(1,10000))            
            return redirect(url_for('DQTestTool.validate', datasetId=datasetId))
        flash(error)

    return render_template('import.html')

@bp.route('/validate', methods=["GET","POST"])
def validate():
    db=get_db()
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords", con=db)
    TFdataFrame=pd.read_sql(sql="SELECT * FROM TF", con=db)

    dataCollection=DataCollection()
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1))

    #Prepare Training data by removing actual faults
    datasetId=request.args.get('datasetId')
    truePositive=0.0
    truePositiveRate=0.0
    if request.method == "POST":
     numberOfClusters=request.form["numberOfClusters"]
     if numberOfClusters:
        for  i in request.form.getlist('Group'):
            db.execute('Delete from trainingRecords where '+dataFrame.columns.values[0]+' in (SELECT '+ dataFrame.columns.values[0]+ ' FROM Faulty_records_'+str(i)+')')
            #TFdataFrame: stores all faults
            TFdataFrame=TFdataFrame.append(pd.read_sql('SELECT DISTINCT '+ dataFrame.columns.values[0]+ ' AS fault_id,"'+datasetId+'" AS dataset_id FROM Faulty_records_'+str(i),con=db))
            if not TFdataFrame.empty:
                TFdataFrame.to_sql('TF',con=db,if_exists='replace',index=False)
    
    dataFrameTrain=pd.read_sql(sql="SELECT * FROM trainingRecords", con=db)
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0]], axis=1))
    
    #Tune and Train model
    autoencoder = Autoencoder()
    hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
    l2Opt = [1e-4,1e-2]
    hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
    bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=200,standardize = True,categorical_encoding='auto'),dataFrameTrainPreprocessed)

    #Assign invalidity scores
    invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)

    #Assign invalidity scores per feature
    invalidityScoresPerFeature=autoencoder.assignInvalidityScorePerFeature(bestModel, dataFramePreprocessed)
    invalidityScoresPerFeature= pd.concat([dataFrame[dataFrame.columns[0]], invalidityScoresPerFeature], axis=1, sort=False)
    invalidityScoresPerFeature.to_sql('Invalidity_scores_per_feature', con=db, if_exists='replace', index=False) 

    #Detect faulty records
    testing=Testing()  
    faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores,sum(invalidityScores)/len(invalidityScores))#np.percentile(invalidityScores,0.5))
    
    #Detect normal records
    normalRecordFrame=testing.detectNormalRecords(dataFrame,invalidityScores,sum(invalidityScores)/len(invalidityScores)) #invalidityScores,np.percentile(invalidityScores,0.5))
    #TODO: Update threshold
    
    #store all the detected faulty records in db
    faultyRecordFrame.to_sql('Faulty_records_all', con=db, if_exists='replace', index=False)


    #store TPR in database for this run
    if not TFdataFrame.empty:
        truePositiveRate=float(len(TFdataFrame['fault_id'].unique().tolist()))/float(faultyRecordFrame.shape[0])
        db.execute('INSERT INTO scores (time, dataset_id, true_positive_rate, false_positive_rate) VALUES (?, ?, ?, ?)',(datetime.datetime.now(), datasetId, truePositiveRate, 1-truePositiveRate))

    #Cluster the faulty records
    faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0]],axis=1))   
    som = SOM(5,5, len(faultyRecordFrame.columns.values)-1, 400)
    dataFrames=som.clusterFaultyRecords(faultyRecordFramePreprocessed, faultyRecordFrame)
    
    #Show groups of faulty records as HTML tables
    i=0
    for dataFrame in dataFrames:
        dataFrame.to_sql('Faulty_records_'+str(i), con=db, if_exists='replace', index=False)
        i=i+1
    numberOfClusters=i
    faulty_records_html=[]
    cluster_scores_fig_url=[]
    cluster_dt_url=[]
    cluster_interpretation=[]
    for i in range(int(numberOfClusters)):
        faulty_records=pd.read_sql(sql="SELECT * FROM Faulty_records_"+str(i), con=db)
        faulty_records_html.append(faulty_records.to_html())
        
        #Show descriptive graph for each group
        cluster_scores=pd.read_sql(sql="SELECT * FROM Invalidity_scores_per_feature WHERE "+dataFrame.columns.values[0]+" IN "+"(SELECT "+dataFrame.columns.values[0]+" FROM Faulty_records_"+str(i)+")", con=db)
        X=dataFrame.columns.values[1:-1]
        Y=cluster_scores.mean().tolist()[1:]
        cluster_scores_fig_url.append(dataCollection.build_graph(X,Y))
        
        #Interpret each cluster
        normalRecordFrame['label']=0
        faulty_records['label']=1
        decisionTreeTrainingFrame= pd.concat([normalRecordFrame,faulty_records])
        print decisionTreeTrainingFrame.columns.values
        decisionTreeTrainingFramePreprocessed=dataCollection.preprocess2(decisionTreeTrainingFrame.drop(decisionTreeTrainingFrame.columns[0], axis=1))
        decisionTree=DecisionTree()
        print "#############"
        columnNames=list(decisionTreeTrainingFramePreprocessed.columns.values)
        columnNames.remove('label')
        columnNames.remove('invalidityScore')
        print columnNames
        treeModel=decisionTree.train(decisionTreeTrainingFramePreprocessed,columnNames,'label' )
        cluster_dt_url.append(decisionTree.visualize(treeModel,columnNames,['Normal','Faulty']))
        treeCodeLines=decisionTree.treeToCode(treeModel,columnNames)
        cluster_interpretation.append(decisionTree.interpret(treeCodeLines))
       
        
    if request.method == 'POST':
        knownFaults = request.form.get("knownFaults")
        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId, knownFaults=knownFaults))
         
    return render_template('validate.html', data='@'.join(faulty_records_html),  numberOfClusters=numberOfClusters, fig_urls=cluster_scores_fig_url,cluster_dt_url=cluster_dt_url, cluster_interpretation=cluster_interpretation)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    knownFaults=request.args.get('knownFaults')

    #A
    faulty_records=pd.read_sql(sql="SELECT * FROM Faulty_records_all", con=db)
    A=set(faulty_records[faulty_records.columns.values[0]].astype(str).tolist())
    
    #TF
    TFdataFrame=pd.read_sql(sql="select distinct fault_id from TF where dataset_id like '"+datasetId+"'", con=db )  
    TF=set(TFdataFrame['fault_id'].astype(str).unique().tolist())
    
    #E
    E=set()
    if knownFaults:
        dataCollection=DataCollection()
        E=dataCollection.csvToSet('datasets/PD/'+str(knownFaults))
    

    evaluation=Evaluation()
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    
    
    #statistics    
    TPR=evaluation.truePositiveRate(A,TF)
    TPGR=evaluation.truePositiveGrowthRate(score)
    NR=evaluation.numberOfRuns(score)
    PD=evaluation.previouslyDetectedFaultyRecords(A,E)
    ND=evaluation.newlyDetectedFaultyRecords(A, E, TF)
    UD=evaluation.unDetectedFaultyRecords(A, E)

    #TODO:save

    return render_template('evaluation.html',  score=score.to_html(), TF=float(len(TF)), A=float(len(A)), TPR=TPR, NR=NR, TPGR=TPGR, E=float(len(E)), PD=PD, ND=ND, UD=UD)

