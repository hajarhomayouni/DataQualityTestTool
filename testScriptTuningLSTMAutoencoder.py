from backendClasses.DQTestToolHelper import DQTestToolHelper
#from DataQualityTestTool.db import get_db
#from db import get_db
import sys
from DQTestTool import *
#from __init__ import *
import sqlite3
#inputes: dataRecordsFilePath,trainedModelFilePath,knownFaultsFilePath,constraintDiscoveryMethod
constraintDiscoveryMethod=sys.argv[4]


db=sqlite3.connect("/s/bach/h/proj/etl/shlok/instance/dq.sqlite")
dQTestToolHelper=DQTestToolHelper()
datasetId=dQTestToolHelper.importData(db,dataRecordsFilePath=sys.argv[1],trainedModelFilePath=sys.argv[2],knownFaultsFilePath=sys.argv[3])
#
hyperParameters={}
numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious'",con=db)
numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious'", con=db)
dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)    
AFdataFrameOld=pd.DataFrame(columns=[dataFrame.columns.values[0]])

#Dictionay and Lists of different window sizes
window_size = {"win_size:10" : 10, "win_size:20" : 20, "win_size:30" : 30, "win_size:40" : 40, "win_size:50" : 50}
windows_size_keys = list(window_size.keys())
windows_size_values = list(window_size.values()) 

#
faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,faultyThresholdRecords,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,hyperParameters,win_size=10)    

#Results will store all the F1_T values for different window sizes difined above
results = []
dataset_index=0
for w_size in windows_size_values:
	hyperParameters = {"win_size:" + str(w_size)}
	faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,hyperParameters,win_size=w_size)    

	faultyRecordFrame.to_sql('faultyRecordFrame_'+datasetId, con=db, if_exists='replace', index=False)
	db.execute("Update dataRecords_"+datasetId+" set status='suspicious' where  "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from faultyRecordFrame_"+datasetId+")")
	db.execute("Drop table faultyRecordFrame_"+datasetId)
	
	record = (pd.read_sql(sql="select F1_T from scores where dataset_id like '"+datasetId+"'", con=db))
	record = record["F1_T"][dataset_index]
	results.append(record)
	dataset_index+=1

#print(results)

#Get the Index of the Max F1_T values got from different window size
index = results.index(max(results))
#print(index)
print ()
print ("These are the five results produced by different win_size:\n")
print (pd.read_sql(sql="select * from scores where dataset_id like '"+datasetId+"'", con=db))
print ()
print(str(windows_size_keys[index]) + " produced the best F1_T score which is " + str(results[index]))
#print(windows_size_values[index])

