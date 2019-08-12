from backendClasses.DQTestToolHelper import DQTestToolHelper
#from DataQualityTestTool.db import get_db
#from db import get_db
import sys
from DQTestTool import *
#from __init__ import *
import sqlite3
#inputes: dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath,constraintDiscoveryMethod
constraintDiscoveryMethod=sys.argv[4]


db=sqlite3.connect("/home/hajar_homayouni_healthdatacompas/instance/dq.sqlite")
dQTestToolHelper=DQTestToolHelper()
datasetId=dQTestToolHelper.importData(db,dataRecordsFilePath=sys.argv[1],trainedModelFilePath=sys.argv[2],knownFaultsFilePath=sys.argv[3])
#
numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious'",con=db)
numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious'", con=db)
dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)    
AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])
#

faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame)
faultyRecordFrame.to_sql('faultyRecordFrame_'+datasetId, con=db, if_exists='replace', index=False)
db.execute("Update dataRecords_"+datasetId+" set status='suspicious' where  "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from faultyRecordFrame_"+datasetId+")")
db.execute("Drop table faultyRecordFrame_"+datasetId) 


for i in range(10):
    #
    numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious'",con=db)
    numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
    suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious'", con=db)
    dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)    
    AFdataFrameOld=pd.read_sql(sql="select distinct "+dataFrame.columns.values[0]+" from actualFaults_"+datasetId, con=db)
    #

    db.execute("Update dataRecords_"+datasetId+" set status='actualFaults' where status like 'suspicious' and  "+dataFrame.columns.values[0]+" in (select * from knownFaults_"+datasetId+")")

    db.execute("Update dataRecords_"+datasetId+" set status='valid' where status like 'suspicious'  and "+dataFrame.columns.values[0]+" not in (select * from knownFaults_"+datasetId+")")


    faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame)
    faultyRecordFrame.to_sql('faultyRecordFrame_'+datasetId, con=db, if_exists='replace', index=False)
    db.execute("Update dataRecords_"+datasetId+" set status='suspicious' where  "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from faultyRecordFrame_"+datasetId+")")
    db.execute("Drop table faultyRecordFrame_"+datasetId) 

print pd.read_sql(sql="select * from scores where dataset_id like '"+datasetId+"'", con=db)

scores=pd.read_sql(sql="select * from scores", con=db)
with open("results/scores.csv", 'w') as f:
    scores.to_csv(f)
numberOfClusters,faulty_records_html,cluster_scores_fig_url,cluster_dt_url,cluster_interpretation,treeRules=dQTestToolHelper.faultInterpretation(db,datasetId,constraintDiscoveryMethod,clusteringMethod,interpretationMethod,dataFrame,faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,bestModelFileName)

