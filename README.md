# DataQualityTestTool
This tool is an automated data quality test approach that:<br/> 
(1) Discovers the constraints in data records that must be satisfied and <br/>
(2) Detects faulty records that do not satisfy the discovered constraints <br/>

**Steps:**<br/>

**Clone or download the tool**<br/>
git clone https://github.com/hajarhomayouni/DataQualityTestTool.git <br/>
Or <br/>
Downlod from https://github.com/hajarhomayouni/DataQualityTestTool/archive/master.zip <br/>

**Go to the tool directory**<br/>
cd DataQualityTestTool<br/>

**Install Python**<br/>
apt install python3<br/>
apt install python3-venv<br/>

**Install and activate python vitual enviroment**<br/>
python3 -m venv venv<br/>
. venv/bin/activate<br/>

**Install Java 1.8**<br/>
apt install openjdk-8-jre-headless<br/>
export JAVA_HOME=path_to/jdk1.8.0_181/<br/>
export PATH=$JAVA_HOME/bin:$PATH<br/>

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
apt install graphviz<br/>

**Update H2o jar path**</br>
Download h2o.jar that is compatible with your h2o version <br/>
Update the jar path in backendClasses/H2oRandomForest.py and backendClasses/H2oGradientBoosting.py <br/>
h2o_jar_path= '<path_to_file>/h2o.jar'

**Initialize Database**<br/>
flask init-db<br/>

**Create local folders in the project directory**<br/>
mkdir static<br/>
mkdir static/mojo<br/>
mkdir static/images<br/>
mkdir static/model<br/>
mkdir datasets<br/>
mkdir datasets/PD<br/>

**Set model path to your project directory**<br/>
Update model path in backendClass/DQTestToolHelper.py
db.execute("update hyperparameters_"+str(datasetId)+" 
set trainedModelFilePath='<font color="red"><font color="red">PATH_TO<font/>/DataQualityTestTool"+trainedModelFilePath+"'")

**Setup Flask**<br/>
export FLASK_APP=Project_directory<br/>
export FLASK_ENV=development<br/>

**Run Flask**<br/>
To access the tool locally:<br/>
flask run<br/>
Or<br/>
python -m flask run

To access the tool remotely:<br/>
flask run --host=0.0.0.0<br/>
Or<br/>
python -m flask run --host=0.0.0.0</br>

**Open the tool from browser**</br>
host:5000/DQTestTool/import
