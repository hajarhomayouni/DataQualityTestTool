from random import *
import functools
from backendClasses.DataCollection import DataCollection
from backendClasses.SOM import SOM
from backendClasses.Testing import Testing
import h2o
import numpy as np
from backendClasses.Autoencoder import Autoencoder
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
from DataQualityTestTool.db import get_db
import datetime

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

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
            #todo: assign random name for dataRecords table
            dataFrame.to_sql('dataRecords', con=db, if_exists='replace')
            dataFrame.to_sql('trainingRecords', con=db, if_exists='replace')
            datasetId=dataPath.replace('.csv','#') + str(randint(1,10000))            
            return redirect(url_for('DQTestTool.validate', datasetId=datasetId))
        flash(error)

    return render_template('import.html')

@bp.route('/validate', methods=["GET","POST"])
def validate():
    db=get_db()
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords", con=db)
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
            truePositive+=1
            db.execute('Delete from trainingRecords where '+dataFrame.columns.values[0]+' in (SELECT '+ dataFrame.columns.values[0]+ ' FROM Faulty_records_'+str(i)+')')

        truePositiveRate=truePositive/float(numberOfClusters)
        db.execute('INSERT INTO scores (time, dataset_id, true_positive_rate, false_positive_rate) VALUES (?, ?, ?, ?)',(datetime.datetime.now(), datasetId, truePositiveRate, 1-truePositiveRate))

    dataFrameTrain=pd.read_sql(sql="SELECT * FROM trainingRecords", con=db)
 
    print dataFrameTrain.shape

    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0]], axis=1))

    #TODO: Update threshold

    
    #Tune and Train model
    autoencoder = Autoencoder()
    hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
    l2Opt = [1e-4,1e-2]
    hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
    bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=200),dataFrameTrainPreprocessed)

    #Assign invalidity scores
    invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)

    #Detect faulty records
    testing=Testing()
    faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores,0) #sum(invalidityScores)/len(invalidityScores))

   # print faultyRecordFrame.sort_values(by=['invalidityScore'],ascending=False)
    faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0]],axis=1))


    #Cluster the faulty records
    #Train a 5*5 SOM with 400 iterations
    #Exclude id columnand invalidity score for clustering
    som = SOM(5,5, len(faultyRecordFrame.columns.values)-1, 400)
    dataFrames=som.clusterFaultyRecords(faultyRecordFramePreprocessed, faultyRecordFrame)
    
    #show groups of faulty records as HTML tables
    i=0
    for dataFrame in dataFrames:
        dataFrame.to_sql('Faulty_records_'+str(i), con=db, if_exists='replace')
        i=i+1
    numberOfClusters=i
    faulty_records_html=[]
    for i in range(int(numberOfClusters)):
        faulty_records=pd.read_sql(sql="SELECT * FROM Faulty_records_"+str(i), con=db)
        faulty_records_html.append(faulty_records.to_html())
    if request.method == 'POST':
        if request.form.get('evaluations'):
            return redirect(url_for('DQTestTool.evaluations', datasetId=datasetId))
         
    return render_template('validate.html', data='@'.join(faulty_records_html),  numberOfClusters=numberOfClusters)
     
@bp.route('/evaluations', methods=["GET","POST"])
def evaluations():
    db=get_db()
    datasetId=request.args.get('datasetId')
    #store approach scores
    scoreHtml=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db).to_html()
    return render_template('evaluations.html',  score=scoreHtml)




