#require a dot before name if running with flask####
from .backendClasses.DQTestToolHelper import DQTestToolHelper
from .backendClasses.DataCollection import DataCollection
from .backendClasses.PatternDiscovery import PatternDiscovery
from .backendClasses.SklearnDecisionTree import SklearnDecisionTree
from .backendClasses.SklearnRandomForest import SklearnRandomForest
from .backendClasses.H2oGradientBoosting import H2oGradientBoosting
from .backendClasses.H2oRandomForest import H2oRandomForest
from .backendClasses.H2oKmeans import H2oKmeans
from .backendClasses.SOM import SOM
from .backendClasses.Testing import Testing
from .backendClasses.Autoencoder import Autoencoder
from .backendClasses.Pyod import Pyod
from .db import get_db
from .backendClasses.Evaluation import Evaluation
#####################################################
import datetime
import os
from random import *
import functools
import h2o
import numpy as np
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd

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
    hyperParameters={'hidden': [100], 'epochs': 5}
    numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious%'",con=db)
    numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
    suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious%'", con=db)
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)    
    AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])
    TP_T=0.0
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
                    TP_T+=1.0
                    
                else:
                    db.execute("Update dataRecords_"+datasetId+" set  status='valid' where status='suspicious_"+str(i)+"'")
                    #db.execute("Update dataRecords_"+datasetId+" set  status='clean' where status='suspicious_"+str(i)+"'")
        
    
    faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,hyperParameters,TP_T)    
    numberOfClusters=0
    faulty_records_html=[]
    cluster_scores_fig_url=[]
    cluster_dt_url=[]
    cluster_interpretation=[]
    treeRules=[] 
    if constraintDiscoveryMethod=="LSTMAutoencoder":
        numberOfClusters,faulty_records_html,cluster_scores_fig_url,cluster_dt_url,cluster_interpretation,treeRules=dQTestToolHelper.faultyTimeseriesInterpretation(db,interpretationMethod,datasetId,dataFramePreprocessed,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFrameTimeseries,y)
    else:
        numberOfClusters,faulty_records_html,cluster_scores_fig_url,cluster_dt_url,cluster_interpretation,treeRules=dQTestToolHelper.faultInterpretation(db,datasetId,constraintDiscoveryMethod,clusteringMethod,interpretationMethod,dataFrame,faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold)
    db.commit()
    db.close()
    return render_template('validate.html', data='@'.join(faulty_records_html), datasetId=datasetId, numberOfClusters=numberOfClusters, fig_urls=cluster_scores_fig_url,cluster_dt_url=cluster_dt_url, cluster_interpretation=cluster_interpretation, treeRules=treeRules)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    evaluation=Evaluation()
    score=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db)    
    print ("****score")
    print(score)
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


