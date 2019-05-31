# DataQualityTestTool
This tool is an automated data quality test approach that:<br/> 
(1) Discovers the constraints in data records that must be satisfied and <br/>
(2) Detects faulty records that do not satisfy the discovered constraints <br/>

**Requierments:**<br/>

**Install Python 2.7**<br/>
sudo apt install python2.7 python-pip<br/>

**Install and activate python vitual enviroment**<br/>
virtualenv venv<br/>
. venv/bin/activate<br/>

**Install Java 1.8**<br/>
apt install openjdk-8-jre-headless<br/>
export JAVA_HOME=path_to/jdk1.8.0_181/<br/>
export PATH=$JAVA_HOME/bin:$PATH<br/>

**Install H2O**<br/>
pip install h2o<br/>

**Update H2o jar path**</br>
Download h2o.jar and update the jar path in backendClasses/H2oRandomForest and backendClasses/H2oGradientBoosting
h2o_jar_path= '<path_to_file>/h2o.jar'

**Install Tensorflow**<br/>
pip install --upgrade tensorflow<br/>

**Install Flask**<br/>
pip install flask<br/>

**Initialize Database**<br/>
flask init-db<br/>

**Create local folders in the project directory**<br/>
mkdir static<br/>
mkdir static/mojo<br/>
mkdir static/images<br/>
mkdir static/model<br/>
mkdir datasets<br/>
mkdir datasets/PD<br/>

**Setup Flask**<br/>
export FLASK_APP=Project_directory<br/>
export FLASK_ENV=development<br/>
flask run --host=0.0.0.0<br/>
