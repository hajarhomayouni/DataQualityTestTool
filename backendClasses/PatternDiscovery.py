import abc

class PatternDiscovery:

    @abc.abstractmethod
    def tuneAndTrain(self, hyperParameters, model, trainDataFrame, trainedModelFilePath="", y=None):
        pass

    @abc.abstractmethod
    def assignInvalidityScore(self, model, testDataFrame):
        pass

    @abc.abstractmethod
    def assignInvalidityScorePerFeature(self, model, testDataFrame):
        pass
        

    
