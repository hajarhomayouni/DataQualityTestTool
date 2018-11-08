import functools
from DataCollection import DataCollection
from SOM import SOM
from Testing import Testing
import h2o
import numpy as np
from Autoencoder import Autoencoder
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import pandas as pd
#from sqlalchemy import create_engine
#engine = create_engine('sqlite://', echo=False)
from DataQualityTestTool.db import get_db


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
            


            return redirect(url_for('DQTestTool.validate'))
        flash(error)

    return render_template('import.html')

@bp.route('/validate', methods=["GET","POST"])
def validate():
    db=get_db()
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords", con=db)
    dataCollection=DataCollection()
    dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1),['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])

    ####
    dataFrameTrain=dataFrame
    numberOfClusters=request.args.get('numberOfClusters')
    print numberOfClusters
    if numberOfClusters:
        for i in range(int(numberOfClusters)):
            if request.form.get('Group_'+str(i)):
                dataToDrop=pd.read_sql(sql="SELECT * FROM Faulty_records_"+str(i), con=db)
                dataFrameTrain=dataFrameTrain.drop(index=[dataFrame.columns.values[0]])
    #print dataFrameTrain
    dataFrameTrainPreprocessed=dataCollection.preprocess(dataFrameTrain.drop([dataFrameTrain.columns.values[0]], axis=1),['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])
    ####
    
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

    # print faultyRecordFrame.sort_values(by=['invalidityScore'],ascending=False)

    faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0],'invalidityScore'],axis=1), ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high']) 


    #Cluster the faulty records
    #Train a 5*5 SOM with 100 iterations
    #Exclude id columnand invalidity score for clustering
    som = SOM(5,5, len(faultyRecordFrame.columns.values)-2, 400)
    dataFrames=som.clusterFaultyRecords(faultyRecordFramePreprocessed, faultyRecordFrame)
    #print dataFrames
           
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
     print numberOfClusters
     return redirect(url_for('DQTestTool.validate',numberOfClusters=numberOfClusters))
         
    return render_template('validate.html', data='@'.join(faulty_records_html))
     

