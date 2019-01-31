from Interpretation import Interpretation
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import random
import subprocess
from graphviz import Source
from DataCollection import DataCollection


class H2oRandomForest(Interpretation):
    @staticmethod
    def train(trainDataFrame, featuresList, target):
        h2o.init()
        fl1=list(featuresList)
        fl=fl1+[target]
        trainData=trainDataFrame[fl]   
        trainDataHex=h2o.H2OFrame(trainData)
        y=target
        trainDataHex[y] = trainDataHex[y].asfactor()
        dc=DataCollection()
        categoricalColumns=dc.findCategorical(trainDataFrame[featuresList])
        trainDataHex[categoricalColumns] = trainDataHex[categoricalColumns].asfactor()
        model=H2ORandomForestEstimator(model_id="rf_covType_v2",ntrees=1, max_depth=4,mtries=len(featuresList)-1)
        model.train(y=y, x=list(featuresList),training_frame=trainDataHex)
        return model
    
    def visualize(self,model,featuresList, targetValues):
        randName=str(random.randint(1,100000000))
        mojo_file_name = "./static/mojo/mojo_"+randName+".zip"
        h2o_jar_path= './venv/lib/python2.7/site-packages/h2o/backend/bin/h2o.jar'
        mojo_full_path = mojo_file_name
        gv_file_path = "./static/mojo/gv_"+randName+".gv"
        image_file_path="./static/images/img_"+randName+".png"
        model.download_mojo(mojo_file_name)
        self.generateGraphviz(h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id = 0)
        self.generateTreeImage(gv_file_path, image_file_path, 0)
        return "/static/images/img_"+randName+".png"


    def generateGraphviz(self,h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id = 0):
        result = subprocess.call(["java", "-cp", h2o_jar_path, "hex.genmodel.tools.PrintMojo", "--tree", str(tree_id), "-i", mojo_full_path , "-o", gv_file_path ], shell=False)
        result = subprocess.call(["ls",gv_file_path], shell = False)
        if result is 0:
            print("Success: Graphviz file " + gv_file_path + " is generated.")
        else: 
            print("Error: Graphviz file " + gv_file_path + " could not be generated.")

    def generateTreeImage(self,gv_file_path, image_file_path, tree_id):
        result = subprocess.call(["dot", "-Tpng", gv_file_path, "-o", image_file_path], shell=False)
        result = subprocess.call(["ls",image_file_path], shell = False)
        if result is 0:
            print("Success: Image File " + image_file_path + " is generated.")
            print("Now you can execute the follow line as-it-is to see the tree graph:") 
            print("Image(filename='" + image_file_path + "\')")
        else:
            print("Error: Image file " + image_file_path + " could not be generated.")



    @staticmethod
    def treeToCode(treeModel, featuresList):
       return None
   
    @staticmethod 
    def interpret(treeCodeLines):
        return None
                    
    @staticmethod
    def treeToRules(treeModel, featuresList):
        return None
