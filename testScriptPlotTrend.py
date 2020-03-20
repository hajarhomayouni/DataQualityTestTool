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

#Reading the Scores CSV File and converting it to SQL Table for better reading
db=sqlite3.connect("scores.sqlite")
cursor = db.cursor()
cursor.execute("DROP TABLE scores_sq")
data = pd.read_csv(fileName).to_sql('scores_sq', con=db);

#Readinf the ScoreID CSV File
iddata = pd.read_csv(idName, header=0)
iddataframe = DataFrame(iddata, columns=["id"])
iddataframe = iddataframe["id"].values.tolist()

#print(pd.read_sql(sql="select * from scores_sq", con=db));
#print(pd.read_sql(sql="select dataset_id,time,HP from scores_sq", con=db))
id_win_size = []

#Gets the window size/Hyper Parameter for all the datasets id passed in the scoreid.csv file
def get_win_size(iddataframe):
    for id in iddataframe:
        wsize = pd.read_sql(sql="select HP from scores_sq where dataset_id like '" + id + "'", con=db)
        wsize = wsize["HP"][0]
        id_win_size.append(wsize)

get_win_size(iddataframe)

time_taken_by_datasets = []

#Calcuates the time difference for the datasets id passed in the scoreid.csv file
def calculate_time(iddataframe ,time_taken_by_datasets):
    for id in iddataframe:
        time = pd.read_sql(sql="select time from scores_sq where dataset_id like '" + id + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + id + "'")
        total_runs = cursor.fetchone()[0]

        stime = time["time"][0]
        etime = time["time"][total_runs-1]
        starttime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S.%f")
        endtime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S.%f")
        timediff = endtime - starttime
        stringtime = str(timediff.seconds) + "." + str(timediff.microseconds)
        time_taken_by_datasets.append(stringtime)

calculate_time(iddataframe, time_taken_by_datasets)

growth_rates = []

def calculate_growth_rate(iddataframe, growth_rates):
    for id in iddataframe:
        F1_Ts = pd.read_sql(sql="select F1_T from scores_sq where dataset_id like '" + id + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + id + "'")
        NR = cursor.fetchone()[0] - 1

        F1_T1 = F1_Ts["F1_T"][0]
        F1_TNR = F1_Ts["F1_T"][NR]

        F1_TGR = (F1_TNR/F1_T1)**(1/NR) - 1

        growth_rates.append(F1_TGR)

calculate_growth_rate(iddataframe, growth_rates)

#Method to create graphs from all DataSets for a specific Column
def plot_by_column(Name):
    fig=plt.figure(figsize=(20,10))
    plt.xlabel("Number of Tests", fontsize=20)
    plt.ylabel(Name, fontsize=20)
    
    plot_title = iddataframe[0][0:iddataframe[0].rfind('_')]

    plt.title(plot_title, fontsize=30)
    markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
    textstr = ""
    xcorrdinate = 0

    for i in range(0, len(iddataframe)):
        
        column_data = pd.read_sql(sql="select " + Name + " from scores_sq where dataset_id like '" + iddataframe[i] + "'", con=db)
        cursor.execute("SELECT COUNT(*) FROM scores_sq WHERE dataset_id LIKE '" + iddataframe[i] + "'")
        total_runs = cursor.fetchone()[0]

        graph_data = []

        for j in range(0, total_runs):
            graph_data.append(column_data[Name][j])

        xaxis = list(range(1,total_runs+1))
        xcorrdinate = xaxis[len(xaxis)-1]+0.6
        textstr = textstr + id_win_size[i] + ":\n" + "Time Taken: " + time_taken_by_datasets[i] + " seconds.\nF1_TGR: " + str(round(growth_rates[i], 4)) + "\n\n";
        plt.plot(xaxis, graph_data, marker=markers[i], markersize=10, label=id_win_size[i])
        plt.xticks(xaxis, fontsize=15)
        plt.yticks(fontsize=15)
    
    plt.ylim(bottom=0)
    plt.text(xcorrdinate,0,textstr, fontsize=20)
    plt.legend(loc="upper left", prop={"size":20}, bbox_to_anchor=(1, 0.7))
    plt.tight_layout(pad=8)
    fig.savefig(plot_title + "_" + Name + ".png", dpi=fig.dpi)

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