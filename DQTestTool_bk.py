import os
from backendClasses.DQTestToolHelper import DQTestToolHelper
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
#from DataQualityTestTool.db import get_db
from db import get_db
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
            db=get_db()
            dQTestToolHelper=DQTestToolHelper()
            datasetId=dQTestToolHelper.importData(db,dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath)
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
    dQTestToolHelper=DQTestToolHelper()
    dataCollection=DataCollection()
    testing=Testing()
    #
    numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious%'",con=db)
    numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
    suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious%'", con=db)
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)    
    AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])
    #
    if request.method == "POST":
        #select actual faluts from previous run before updating the database - we need this information to measure the false negative rate
        AFdataFrameOld=pd.read_sql(sql="select distinct "+dataFrame.columns.values[0]+" from actualFaults_"+datasetId, con=db)

        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId))
        
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #@@@@@@@@@@@@@@Incorporate domain knowledge@@@@@@@@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        numberOfClusters=request.form["numberOfClusters"]
        #maxInvalidityScoreOfNormalData=[]
        if numberOfClusters:
            for i in range(int(numberOfClusters)):

                if str(i) in request.form.getlist('Group'):
                    db.execute("Update dataRecords_"+datasetId+" set  status='actualFaults_"+str(i)+ "' where status='suspicious_"+str(i)+"'")
                    
                else:
                    db.execute("Update dataRecords_"+datasetId+" set  status='valid' where status='suspicious_"+str(i)+"'")
                    #db.execute("Update dataRecords_"+datasetId+" set  status='clean' where status='suspicious_"+str(i)+"'")

    faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame)    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@Cluster suspicious records@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #If you want to work with data directly for clustering, use faultyRecordFrame directly. Now it clusters based on invelidity score per feature
    dataFrames=[]
    if clusteringMethod=="som":
        #Detect faulty records based on invalidity scores
        faultyInvalidityScoreFrame=testing.detectFaultyRecords(invalidityScoresPerFeature,invalidityScores,faultyThreshold)#,statistics.mean(invalidityScores))#,np.percentile(invalidityScores,0.5))
        faultyInvalidityScoreFrame.columns=dataFrame.columns.values[:-1]
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


