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

#from dq.db import get_db

bp = Blueprint('DQTestTool', __name__, url_prefix='/DQTestTool')


@bp.route('/import', methods=["GET","POST"])
def importDataFrame():
    """
    Import CSV data 
    """
    if request.method == 'POST':
        dataPath = request.form['dataPath']
        error = None

        if not dataPath:
            error = 'Data path is required.'

        if error is None:
            dataCollection=DataCollection()
            dataFrame=dataCollection.importData(dataPath).head(50)
            db=get_db()
            #todo: assign random name for dataRecords table
            dataFrame.to_sql('dataRecords', con=db, if_exists='replace')
            dataFrame.to_sql('trainingRecords', con=db, if_exists='replace')
            
            return redirect(url_for('DQTestTool.validate'))
        flash(error)

    return render_template('import.html')

@bp.route('/validate', methods=["GET","POST"])
def validate():
    db=get_db()
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords", con=db)
    dataCollection=DataCollection()
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1),['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])

    #Prepare Training data by removing actual faults
    datasetId="random"
    truePositive=0.0
    truePositiveRate=0.0
    if request.method == "POST":
     numberOfClusters=request.form["numberOfClusters"]
     if numberOfClusters:
        for  i in request.form.getlist('Group'):
            truePositive+=1
            #dataFrameTrain=pd.read_sql(sql="SELECT * FROM trainingRecords where "+dataFrameTrain.columns.values[1]+" not in ( SELECT "+dataFrameTrain.columns.values[1]+" FROM Faulty_records_"+str(i)+")", con=db)
            db.execute('Delete from trainingRecords where '+dataFrame.columns.values[0]+' in (SELECT '+ dataFrame.columns.values[0]+ ' FROM Faulty_records_'+str(i)+')')
            #dataFrameTrain.to_sql('trainingRecords', con=db, if_exists='replace')

        truePositiveRate=truePositive/float(numberOfClusters)
        db.execute('INSERT INTO scores (time, dataset_id, true_positive_rate, false_positive_rate) VALUES (?, ?, ?, ?)',(datetime.datetime.now(), datasetId, truePositiveRate, 1-truePositiveRate))

    dataFrameTrain=pd.read_sql(sql="SELECT * FROM trainingRecords", con=db)
    print dataFrameTrain.shape
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0]], axis=1),['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])

    #TODO: Update threshold

    #store approach scores
    scoreHtml=pd.read_sql(sql="SELECT * FROM scores where dataset_id like '"+datasetId+"'", con=db).to_html()
    
    #Tune and Train model
    autoencoder = Autoencoder()
    hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
    l2Opt = [1e-4,1e-2]
    hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
    bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=100),dataFrameTrainPreprocessed)

    #Assign invalidity scores
    invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)

    #Detect faulty records
    testing=Testing()
    faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores, sum(invalidityScores)/len(invalidityScores))

    #print faultyRecordFrame.sort_values(by=['invalidityScore'],ascending=False)
    faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0],'invalidityScore'],axis=1), ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high']) 


    #Cluster the faulty records
    #Train a 5*5 SOM with 400 iterations
    #Exclude id columnand invalidity score for clustering
    som = SOM(5,5, len(faultyRecordFrame.columns.values)-2, 400)
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
     return redirect(url_for('DQTestTool.validate'))
         
    return render_template('validate.html', data='@'.join(faulty_records_html), score=scoreHtml, numberOfClusters=numberOfClusters)
     

