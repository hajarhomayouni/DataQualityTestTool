#!/bin/bash
#FILES=./datasets/*
for f in "lymphography.csv"
do
  echo "NEW FIlE ***************************************************************************************"
  echo $f
  if [ ${f: -4} == ".csv" ]
  then
  	python testScriptTuningAutoencoder.py "datasets/${f}" "" "datasets/PD/${f::-4}_outliers.csv" "H2O_Autoencoder" 
  fi
done
