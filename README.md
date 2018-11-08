# DataQualityTestTool
This tool is an automated data quality test approach that:<br/> 
(1) Discovers the constraints in data records that must be satisfied and <br/>
(2) Detects faulty records that do not satisfy the discovered constraints <br/>

Requierments:<br/>

Python 2.7<br/>
sudo apt install python2.7 python-pip<br/>

Java 1.8<br/>
apt install openjdk-8-jre-headless<br/>
export JAVA_HOME=path_to/jdk1.8.0_181/<br/>
export PATH=$JAVA_HOME/bin:$PATH<br/>

H2O<br/>
pip install h2o<br/>

Tensorflow<br/>
pip install --upgrade tensorflow<br/>

Install and activate python vitual enviroment<br/>
virtualenv venv<br/>
. venv/bin/activate<br/>

Install flask<br/>
pip install flask<br/>

Setup flask:<br/>
export FLASK_APP=Project_directory<br/>
export FLASK_ENV=development<br/>
flask run --host=0.0.0.0<br/>
