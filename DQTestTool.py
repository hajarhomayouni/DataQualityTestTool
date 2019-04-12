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
        else:
            error="No data file is selected"
        if error is None:
            dataCollection=DataCollection()
            dataFrame=dataCollection.importData("datasets/"+dataRecordsFile.filename).head(50)
            db=get_db()
            #all the data records are clean by default
            dataFrame['status']='clean'
            datasetId=dataRecordsFile.filename.replace('.csv','_').replace("-","_") + str(randint(1,10000))
            dataFrame.to_sql('dataRecords_'+datasetId, con=db, if_exists='replace')           
            return redirect(url_for('DQTestTool.validate',trainedModelFilePath=trainedModelFilePath, datasetId=datasetId, interpretationMethod=interpretationMethod, clusteringMethod=clusteringMethod))
        flash(error)

    return render_template('import.html')


@bp.route('/validate', methods=["GET","POST"])
def validate():
    data='fisrt load'
       
    if request.method == 'POST':
        if request.form.get('revalidate'):
            data='second load'
            return redirect(url_for('DQTestTool.validate'))
        if request.form.get('evaluation'):
            return redirect(url_for('DQTestTool.evaluation', datasetId=datasetId, knownFaults=knownFaults))

    return render_template('validate.html', data=data)
     
@bp.route('/evaluation', methods=["GET","POST"])
def evaluation():
    db=get_db()
    datasetId=request.args.get('datasetId')
    knownFaults=request.args.get('knownFaults')

    #A
    suspiciousRecords=pd.read_sql(sql="SELECT distinct * FROM dataRecords_"+datasetId+" where status like 'suspicious_%' or status like 'actualFaults_%'", con=db)
    A=set(suspiciousRecords[suspiciousRecords.columns.values[0]].astype(str).tolist())
    
    #AF
    actualFaults=pd.read_sql(sql="select distinct * from dataRecords_"+datasetId+" where status like 'actualFaults_%'", con=db )  
    AF=set(actualFaults[actualFaults.columns.values[0]].astype(str).unique().tolist())
    
    #E
    E=set()
    if knownFaults:
        dataCollection=DataCollection()
        E=dataCollection.csvToSet(str(knownFaults))
    

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

