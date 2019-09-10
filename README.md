# DataQualityTestTool
This tool is an automated data quality test approach that:<br/> 
(1) Discovers the constraints in data records that must be satisfied and <br/>
(2) Detects faulty records that do not satisfy the discovered constraints <br/>

## Requirements
 python3 and java1.8
 
## Steps to install the tool
**Clone or download the tool**<br/>
git clone https://github.com/hajarhomayouni/DataQualityTestTool.git <br/>
*Or* <br/>
Download from https://github.com/hajarhomayouni/DataQualityTestTool/archive/master.zip <br/>

**Go to the tool directory**<br/>
cd DataQualityTestTool<br/>

**Install and activate python virtual environment**<br/>
*On Linux* <br/>
python3 -m venv venv<br/>
. venv/bin/activate<br/>

*On Mac* <br/>
mkdir venv <br/>
Python3 -m vena ./venv <br/>
Source vent/bin/activate <br/>

**Install python packages**<br/>
pip install h2o<br/>
pip install IPython<br/>
pip install pydot<br/>
pip install matplotlib<br/>
pip install pandas<br/>
pip install statistics<br/>
pip install  kmodes<br/>
pip install pyod<br/>
pip install keras<br/>
pip install --upgrade tensorflow<br/>
pip install flask<br/>

**Install Graphviz**<br/>
*On Linux Ubuntu*<br/>
apt install graphviz<br/>

*On Mac*<br/>
brew install graphviz<br/>

**Update H2o jar path**</br>
Download h2o.jar that is compatible with your h2o version <br/>
Update the jar path in backendClasses/H2oRandomForest.py and backendClasses/H2oGradientBoosting.py: <br/>
h2o_jar_path= '<path_to_file>/h2o.jar'

**Create local folders in the project directory**<br/>
mkdir static<br/>
mkdir static/mojo<br/>
mkdir static/images<br/>
mkdir static/model<br/>
mkdir datasets<br/>
mkdir datasets/PD<br/>

**Set model path to your project directory**<br/>
Update model path in backendClass/DQTestToolHelper.py: <br/>
db.execute("update hyperparameters_"+str(datasetId)+" 
set trainedModelFilePath='<font color="red">PATH_TO<font/>/DataQualityTestTool"+trainedModelFilePath+"'")<br/>

**Setup Flask**<br/>
export FLASK_APP=Project_directory<br/>
export FLASK_ENV=development<br/>

**Initialize Database**<br/>
flask init-db<br/>

**Run Flask**<br/>
*To access the tool locally*:<br/>
flask run<br/>
*Or*<br/>
python -m flask run

*To access the tool remotely*:<br/>
flask run --host=0.0.0.0<br/>
*Or*<br/>
python -m flask run --host=0.0.0.0</br>

**Open the tool from browser**</br>
host:5000/DQTestTool/import

## Steps to run the tool in terminal

**Run the testScript.py with appropriate arguments**<br/>
*Run the following command inside the project directory:*<br/>
python testScript.py dataRecordsFilePath, trainedModelFilePath, knownFaultsFilePath, constraintDiscoveryMethod <br/>

where<br/>
dataRecordsFilePath should be set to the full path to the data in form of CSV </br>
trainedModelFilePath should be set to "" unless you want to use a previously trained model </br>
knownFaultsFilePath should be set to the full path to the CSV file that stores IDs of previously known faulty records. Set to "" if there is no previously known faults </br>
constraintDiscoveryMethod should be set to "H2o_Autoencoder"<br/>

**Note:** The first column of your CSV data file should be a unique ID </br>

**See the output in the scores.csv file stored in the results folder**<br/>
results/scores.csv<br/>





