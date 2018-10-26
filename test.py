from DataCollection import DataCollection


#work with classes
dataCollection= DataCollection()
dataFrame=dataCollection.selectFeatures(1)
print dataFrame


#Read csv data
dataFrame=dataCollection.importData("testData.csv")
print dataCollection.preprocess(dataFrame, ['measurement_concept_id','gender_concept_id'], ['year_of_birth'])
