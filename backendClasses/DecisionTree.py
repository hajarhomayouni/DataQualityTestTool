from sklearn import tree
import graphviz 

class DecisionTree:

    @staticmethod
    def trainTree(trainDataFrame, featuresList, target):
        dt = tree.DecisionTreeClassifier()
        return dt.fit(trainDataFrame[featuresList], trainDataFrame[target])

    
        
    @staticmethod
    def visualizeTree(treeModel, trainDataFrame, featuresList, targetValues):
        dot_data = tree.export_graphviz(treeModel, out_file=None,
                      feature_names=featuresList,  
                      class_names=targetValues,  
                      filled=True, rounded=True,  
                      special_characters=True)
        graph = graphviz.Source(dot_data)
        retrun graph
        
        
