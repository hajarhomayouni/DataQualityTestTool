import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator


class GradientBoosting:
    @staticmethod
    def train(trainDataFrame, featuresList, target):
        h2o.init()
        fl1=list(featuresList)
        fl=fl1+[target]
        trainData=trainDataFrame[fl]   
        trainDataHex=h2o.H2OFrame(trainData)
        print trainDataHex
        y=target
        trainDataHex[y] = trainDataHex[y].asfactor()
        model=H2OGradientBoostingEstimator(distribution="bernoulli",ntrees=10, max_depth=4,learn_rate=0.1)
        model.train(y=y, x=list(featuresList),training_frame=trainDataHex)
        return model

    @staticmethod
    def visualize(model):
        modelfile = model.download_mojo(path="./mojo/", get_genmodel_jar=True)
        mojo_file_name = "./mojo/my_gbm_mojo.zip"
        h2o_jar_path= '/Users/avkashchauhan/tools/h2o-3/h2o-3.14.0.3/h2o.jar'
        mojo_full_path = mojo_file_name
        gv_file_path = "./mojo/gv"+randint()+".gv"
        generateGraphviz(h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id = 0)
        graph = Source(gv_file_path)
        graph_png=graph.pipe(format='png')
        graph_url=base64.b64encode(graph_png).decode('utf-8')
        return 'data:image/png;base64,{}'.format(graph_url)


    @staticmethod
    def generateGraphviz(h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id = 0):
        image_file_path = image_file_path + "_" + str(tree_id) + ".png"
        result = subprocess.call(["java", "-cp", h2o_jar_path, "hex.genmodel.tools.PrintMojo", "--tree", str(tree_id), "-i", mojo_full_path , "-o", gv_file_path ], shell=False)
        result = subprocess.call(["ls",gv_file_path], shell = False)
        if result is 0:
            print("Success: Graphviz file " + gv_file_path + " is generated.")
        else: 
            print("Error: Graphviz file " + gv_file_path + " could not be generated.")

