#Note: This script only works for H2O_Autoencoder. Updates need to be made in backend classes to make it work for LSTMAutoencoder

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
numberOfSuspiciousDataFrame=pd.read_sql(sql="select count(*) from dataRecords_"+datasetId+ " where status like 'suspicious'",con=db)
numberOfSuspicious=numberOfSuspiciousDataFrame[numberOfSuspiciousDataFrame.columns.values[0]].values[0]
suspiciousDataFrame=pd.read_sql(sql="select * from dataRecords_"+datasetId+" where status like 'suspicious'", con=db)
dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)
print(dataFrame)
AFdataFrameOld=pd.DataFrame(data=[-1],columns=[dataFrame.columns.values[0]])
truePositiveRateGroup=0.0
#
hidden_layers=range(1,4)
hidden_nodes=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
epochs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

dropout_ratios=[0,0.1,0.2]
ls=[0, 1e-2]
#for hiddenOpt in hiddenOpts:
for layer in hidden_layers:
 for node in hidden_nodes:
  for epoch in epochs:
   #for dropout_ratio in dropout_ratios:
    #for l in ls:
        hiddenOpt=[node]*layer
        print(hiddenOpt)
        #hidden_dropout_ratios=[dropout_ratio]*layer
        #input_dropout_ratio=dropout_ratio
        hyperParameters={"hidden":hiddenOpt,"epochs":epoch}#, "hidden_dropout_ratios":hidden_dropout_ratios, "input_dropout_ratio":input_dropout_ratio,"l2":l}
        print(hyperParameters)
        #
        suspiciousDataFrame=pd.DataFrame()
        dataFrame=pd.read_sql(sql="SELECT * FROM dataRecords_"+datasetId, con=db)  
        AFdataFrameOld=pd.DataFrame()
        truePositiveRateGroup=0.0
        
        faultyRecordFrame,normalRecordFrame,invalidityScoresPerFeature,invalidityScores,faultyThreshold,yhatWithInvalidityScores,XWithInvalidityScores,mse_attributes,faultyTimeseriesIndexes,normalTimeseriesIndexes,dataFramePreprocessed,dataFrameTimeseries,y=dQTestToolHelper.constraintDiscoveryAndFaultDetection(db,datasetId,dataFrame,constraintDiscoveryMethod,AFdataFrameOld,suspiciousDataFrame,truePositiveRateGroup,hyperParameters)    
        faultyRecordFrame.to_sql('faultyRecordFrame_'+datasetId, con=db, if_exists='replace', index=False)
        db.execute("Update dataRecords_"+datasetId+" set status='suspicious' where  "+dataFrame.columns.values[0]+" in (select "+dataFrame.columns.values[0]+ " from faultyRecordFrame_"+datasetId+")")
        db.execute("Drop table faultyRecordFrame_"+datasetId) 

print (pd.read_sql(sql="select * from scores where dataset_id like '"+datasetId+"'", con=db))

scores=pd.read_sql(sql="select * from scores", con=db)
with open("results/scores.csv", 'w') as f:
    scores.to_csv(f)

