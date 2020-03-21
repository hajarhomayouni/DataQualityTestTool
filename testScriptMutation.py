#Name: Shlok Gopalbhai Gondalia
#Email: shlok@rams.colostate.edu
#Date: Friday 20, March

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import csv
import sqlite3
from datetime import datetime

#Reading the name of file from User
fileName = input("Input File Name in CSV Format:\n");

#Check to see whether the fileName is correct or not
while True:
    if(fileName.endswith(".csv")):
        #print(fileName)
        break
    else:
        print("Invalid File Format, Please write the File Name in .csv Format")
        fileName = input("Enter File Name Again:\n");

#print(fileName)
idName = input("Input ID File Name in CSV Format:\n");

#Check to see whether the fileName is correct or not
while True:
    if(idName.endswith(".csv")):
        #print(idName)
        break
    else:
        print("Invalid ID File Format, Please write the File Name in .csv Format")
        idName = input("Enter File Name Again:\n");

#Reading the column number so that it is easy to find particular column
columnNum = "-1"
def column_number():
    number = input("""Write a number for specific column:
0: All Columns
1: Loss
2: PD
3: SD
4: F1
5: UD
6: ND
7: FPR
8: TPR
9: TPR_T
10: FPR_T
11: F1_T\n""");
    return number

columnNum = column_number()

#Check to see whether the columnNum is correct or not
while True:
    if(columnNum == "0" or columnNum == "1" or columnNum == "2" or columnNum == "3" or columnNum == "4" or columnNum == "5" or columnNum == "6" or columnNum == "7" or columnNum == "8" or columnNum == "9" or columnNum == "10" or columnNum == "11"):
        #print(columnNum)
        break
    else:
        print("Invalid Column Number, Please write the correct Column Number")
        columnNum = column_number();

#Finding the corresponding column thorugh Column Name
columnList = ["all_columns", "Loss", "PD", "SD", "F1", "UD", "ND", "FPR", "TPR", "TPR_T", "FPR_T", "F1_T"]

columnName = columnList[int(columnNum)]
print(columnName)

#GraphTitle
graph_title = input("Please Enter the Title for the Plot: (This name will also be used in Filenaming)\n")

#print(graph_title)

#Datasets Type
dataset_type = input("Please Enter the Type of Dataset for the Plot: (e.g. M1)\n")

#print(dataset_type)

#Reading the Scores CSV File and converting it to SQL Table for better reading
db=sqlite3.connect("scores.sqlite")
cursor = db.cursor()
cursor.execute("DROP TABLE IF EXISTS scores_sq")
data = pd.read_csv(fileName).to_sql('scores_sq', con=db);

#Readinf the ScoreID CSV File
iddata = pd.read_csv(idName, header=0)
iddataframewithwinsize = DataFrame(iddata, columns=["id_with_win_size"])
iddataframewithwinsize = iddataframewithwinsize["id_with_win_size"].values.tolist()

iddataframewithautowinsize = DataFrame(iddata, columns=["id_with_auto_win_size"])
iddataframewithautowinsize = iddataframewithautowinsize["id_with_auto_win_size"].values.tolist()

#Calcuates the average time difference for the datasets id passed in the scoreid.csv file
def calculate_time(dataframe, avg_time_taken):
    timetaken = []
    for id in dataframe:
        time = pd.read_sql(sql="select time from scores_sq where dataset_id like '" + id + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + id + "'")
        total_runs = cursor.fetchone()[0]

        stime = time["time"][0]
        etime = time["time"][total_runs-1]
        starttime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S.%f")
        endtime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S.%f")
        timediff = endtime - starttime
        stringtime = timediff.total_seconds()
        timetaken.append(stringtime)

    avg_time = sum(timetaken)/len(timetaken)
    avg_time_taken.append(str(avg_time))

#Calcuates the average growth rate for the datasets id passed in the scoreid.csv file
def calculate_growth_rate(iddataframe, avg_growth_rates):
    growthrate = []
    for id in iddataframe:
        F1_Ts = pd.read_sql(sql="select F1_T from scores_sq where dataset_id like '" + id + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + id + "'")
        NR = cursor.fetchone()[0] - 1

        F1_T1 = F1_Ts["F1_T"][0]
        F1_TNR = F1_Ts["F1_T"][NR]

        F1_TGR = ((F1_TNR/F1_T1)**(1/NR)) - 1

        growthrate.append(F1_TGR)

    avg_growth = sum(growthrate)/len(growthrate)
    avg_growth_rates.append(str(avg_growth))

#Groups the values of results by the test runs
def collector(dataframe, datalist, columnName):
    for i in range(0, len(dataframe)):
        if str(dataframe[i]) == 'nan':
            continue
        
        column_data = pd.read_sql(sql="select " + columnName + " from scores_sq where dataset_id like '" + dataframe[i] + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + dataframe[i] + "'")
        total_runs = cursor.fetchone()[0]

        for j in range(0, total_runs):
            if (j+1) > len(datalist):
                datalist.append([])
            
            data = column_data[columnName][j]
            datalist[j].append(data)

#Calculates the average so that it can be represented as a point in a graph
def average(datalist, coordinatelist):
    avgdatalist = []

    for i in range(0, len(datalist)):
        avg_value = round((sum(datalist[i])/len(datalist[i])),2)
        avgdatalist.append(avg_value)

    coordinatelist.append(avgdatalist)

#Creates a CSV file with all the avg times and avg growth rates
def writeCSV(avg_time_taken, avg_growth_rates):
    avg_time_taken.insert(0, "Average Time Taken in Seconds")
    avg_growth_rates.insert(0, "Average Growth Rates")
    with open('time_growthrate.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Attributes/Datasets", dataset_type+"_datasets_best_win_size", dataset_type+"_datasets_auto_win_size"])
        writer.writerow(avg_time_taken)
        writer.writerow(avg_growth_rates)

#Method to create graphs from all DataSets for a specific Column
def plot_by_column(columnName):
    win_size_datalist = []
    auto_win_size_datalist = []
    collector(iddataframewithwinsize, win_size_datalist, columnName)
    collector(iddataframewithautowinsize, auto_win_size_datalist, columnName)

    avgdatalistcoordinates = []
    average(win_size_datalist, avgdatalistcoordinates)
    average(auto_win_size_datalist, avgdatalistcoordinates)

    avg_time_taken = []
    calculate_time(iddataframewithwinsize, avg_time_taken)
    calculate_time(iddataframewithautowinsize, avg_time_taken)

    avg_growth_rates = []
    calculate_growth_rate(iddataframewithwinsize, avg_growth_rates)
    calculate_growth_rate(iddataframewithautowinsize, avg_growth_rates)

    writeCSV(avg_time_taken, avg_growth_rates)

    fig=plt.figure(figsize=(20,12))
    plt.xlabel("Number of Tests", fontsize=20)
    plt.ylabel(columnName, fontsize=20)
    
    plot_title = graph_title

    plt.title(plot_title, fontsize=30)
    markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
    xcorrdinate = 0

    labels = ["Brute force\nwindowing", "Autocorrelation-based\nwindowing"]

    ymax = avgdatalistcoordinates[0][len(avgdatalistcoordinates[0])-1]

    if ymax < avgdatalistcoordinates[1][len(avgdatalistcoordinates[1])-1]:
        ymax = avgdatalistcoordinates[1][len(avgdatalistcoordinates[1])-1]

    for i in range(0, len(avgdatalistcoordinates)):
        xaxis = list(range(1,len(avgdatalistcoordinates[i])+1))
        xcorrdinate = xaxis[len(xaxis)-1]+0.6
        plt.plot(xaxis, avgdatalistcoordinates[i], marker=markers[i], markersize=10, label=labels[i])
        plt.xticks(xaxis, fontsize=15)
        plt.yticks(fontsize=15)
    
    plt.ylim([0, ymax+0.1])
    plt.legend(loc="upper left", prop={"size":20}, bbox_to_anchor=(1, 0.7))
    plt.tight_layout(pad=8)
    fig.savefig(plot_title + "_" + columnName + ".png", dpi=fig.dpi)

#Method to create all graphs of all DataSets in one go
def plot_all():
    for i in range(1, len(columnList)):
        columnName = columnList[i]
        plot_by_column(columnName)

if columnName == "all_columns":
    plot_all()
else:
    plot_by_column(columnName)

db.commit()
db.close()