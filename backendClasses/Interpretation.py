import abc

class Interpretation:

    @abc.abstractmethod
    def train(trainDataFrame, featuresList, target):
        pass
          
        
    @abc.abstractmethod
    def visualize(model, featuresList, targetValues):
        pass

       
    @abc.abstractmethod 
    def interpret(model, featuresList):
        pass
