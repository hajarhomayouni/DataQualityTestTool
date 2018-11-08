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
            dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0]], axis=1),['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high'])
            #Tune and Train model
            autoencoder = Autoencoder()
            hiddenOpt = [[50,50],[100,100], [5,5,5],[50,50,50]]
            l2Opt = [1e-4,1e-2]
            hyperParameters = {"hidden":hiddenOpt, "l2":l2Opt}
            bestModel=autoencoder.tuneAndTrain(hyperParameters,H2OAutoEncoderEstimator(activation="Tanh", ignore_const_cols=False, epochs=100),dataFramePreprocessed)

            #Assign invalidity scores
            invalidityScores=autoencoder.assignInvalidityScore(bestModel, dataFramePreprocessed)

            #Detect faulty records
            testing=Testing()
            faultyRecordFrame=testing.detectFaultyRecords(dataFrame, invalidityScores, np.median(invalidityScores))

            # print faultyRecordFrame.sort_values(by=['invalidityScore'],ascending=False)

            faultyRecordFramePreprocessed=dataCollection.preprocess(faultyRecordFrame.drop([faultyRecordFrame.columns.values[0],'invalidityScore'],axis=1), ['gender_concept_id','measurement_type_concept_id'], ['year_of_birth','value_as_number','range_low','range_high']) 


            #Cluster the faulty records
            #Train a 5*5 SOM with 100 iterations
            #Exclude id columnand invalidity score for clustering
            som = SOM(5,5, len(faultyRecordFrame.columns.values)-2, 400)
            dataFrames=som.clusterFaultyRecords(faultyRecordFramePreprocessed, faultyRecordFrame)
            #print dataFrames
            
            i=0
            db=get_db()
            for dataFrame in dataFrames:

                dataFrame.to_sql('Faulty_records_'+str(i), con=db, if_exists='replace')
                i=i+1

            """dataFramesHtml=[]
            for dataFrame in dataFrames:
                dataFramesHtml.append(dataFrame.to_html())

            print dataFramesHtml"""

            return redirect(url_for('DQTestTool.validate',numberOfClusters=i))
        flash(error)

    return render_template('import.html')

@bp.route('/validate', methods=["GET","POST"])
def validate():
    faulty_records_html=[]
    numberOfClusters=request.args.get('numberOfClusters')
    db=get_db()
    for i in range(int(numberOfClusters)):
        faulty_records=pd.read_sql(sql="SELECT * FROM Faulty_records_"+str(i), con=db)
        faulty_records_html.append(faulty_records.to_html())
    if request.method == 'POST':
     print numberOfClusters
     return redirect(url_for('DQTestTool.revalidate',numberOfClusters=numberOfClusters))
         
    return render_template('validate.html', data='@'.join(faulty_records_html))

@bp.route('/revalidate', methods=["GET","POST"])
def revalidate():
    faulty_records_html=[]
    if request.method=='GET':
     numberOfClusters=request.args.get('numberOfClusters')
     db=get_db()
     for i in range(int(numberOfClusters)):
         if request.args.get('Group_'+str(i))==True:
             faulty_records=pd.read_sql(sql="SELECT * FROM Faulty_records_"+str(i), con=db)
             faulty_records_html.append(faulty_records.to_html())
         
    return render_template('validate.html', data='@'.join(faulty_records_html))




     

