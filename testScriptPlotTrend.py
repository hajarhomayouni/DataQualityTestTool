#Name: Shlok Gopalbhai Gondalia
#Email: shlok@rams.colostate.edu
#Date: Wed 30, Oct

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import csv

#Reading the name of file from User
fileName = input("Input File Name in CSV Format:\n");

#Check to see whether the fileName is correct or not
while True:
    if(fileName.endswith(".csv")):
        print(fileName)
        break
    else:
        print("Invalid File Format, Please write the File Name in .csv Format")
        fileName = input("Enter File Name Again:\n");

#print(fileName)

idName = input("Input ID File Name in CSV Format:\n");

#Check to see whether the fileName is correct or not
while True:
    if(idName.endswith(".csv")):
        print(idName)
        break
    else:
        print("Invalid ID File Format, Please write the File Name in .csv Format")
        idName = input("Enter File Name Again:\n");

#Reading the column number so that it is easy to find particular column
columnNum = input("Write a number for specific column:\n0: All Columns\n1: Previously Detected\n2: Suspicious Detected\n3: Undetected\n4: Newly Detected\n5: True Negative Rate\n6: False Negative Rate\n7: False Positive Rate\n8: True Positive Rate\n");

#Check to see whether the columnNum is correct or not
while True:
    if(columnNum == "0" or columnNum == "1" or columnNum == "2" or columnNum == "3" or columnNum == "4" or columnNum == "5" or columnNum == "6" or columnNum == "7" or columnNum == "8"):
        print(columnNum)
        break
    else:
        print("Invalid Column Number, Please write the correct Column Number")
        columnNum = input("Write a number for specific column:\n0: All Columns\n1: Previously Detected\n2: Suspicious Detected\n3: Undetected\n4: Newly Detected\n5: True Negative Rate\n6: False Negative Rate\n7: False Positive Rate\n8: True Positive Rate\n");

#print(columnNum)

#Finding the corresponding column thorugh Column Name
columnList = ["all_columns", "previously_detected", "suspicious_detected", "undetected", "newly_detected", "true_negative_rate", "false_negative_rate", "false_positive_rate", "true_positive_rate"]
columnName = columnList[int(columnNum)]
print(columnName)

#Creating a list for x-axis which in our case is the number of tests.
xaxis = list(range(1,11))
#print(xaxis)

#Reading the CSV File
data = pd.read_csv(fileName, header=0)
iddata = pd.read_csv(idName, header=0)

#Creating DataFrame for easy access of columns from readed Data
df = DataFrame(data, columns=["dataset_id", "previously_detected", "suspicious_detected", "undetected", "newly_detected", "true_negative_rate", "false_negative_rate", "false_positive_rate", "true_positive_rate"])

iddataframe = DataFrame(iddata, columns=["id"])
iddf = iddataframe["id"].values.tolist()
#print(iddf)
uniqueDataName = []

#Method to create list of known DataSets in the CSV File
def unique(uniqueDataName):
    dataName = df["dataset_id"].values.tolist()
    
    for i in dataName:
        if i not in uniqueDataName:
            uniqueDataName.append(i)

#Create Unique List of DataSets through unique method
unique(uniqueDataName)
#print(uniqueDataName)

#Method to create graphs from all DataSets for a specific Column
def plot_by_column(Name):
    readCol = df[Name].values.tolist()
    #print(readCol)
    combinedCol = []
    list_seperator(readCol, combinedCol)
    #print(combinedCol)    

    fig=plt.figure(figsize=(20,10))
    plt.xlabel("Number of Tests")
    plt.title(Name)
    markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
        
    for i in range(0, len(iddf)):
        if(iddf[i] in uniqueDataName):
            j = uniqueDataName.index(iddf[i])
            plt.plot(xaxis, combinedCol[j], marker=markers[i], markersize=10, label=uniqueDataName[j])
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    plt.legend(loc="upper left", prop={"size":15}, bbox_to_anchor=(1, 0.7))
    plt.tight_layout(pad=7)
    fig.savefig(Name + ".png", dpi=fig.dpi)

#Method to create all graphs of all DataSets in one go
def plot_all():
    for j in range(1, len(columnList)):
        columnName = columnList[j]
        plot_by_column(columnName)    
    
#Method to create list of lists of unique DataSets from combined DataSets
def list_seperator(readCol, combinedCol):
    #combinedCol = []
    individualCol = []

    """for i in readCol:
        if i == 1 or i == 0:
            combinedCol.append(individualCol)
            individualCol = []
        else:
            individualCol.append(i)

    while True:
        try:
            combinedCol.remove([])
        except ValueError:
            break
    """
    for i in range(1, len(readCol), 11):
        individualCol = readCol[i:i+10]
        combinedCol.append(individualCol)
        i=i+1
    
#Calling the function according to user input
if columnName == "all_columns":
    plot_all()
else:
    plot_by_column(columnName)

