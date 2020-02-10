import csv

import ast

with open('scores_diagram.csv', mode='w') as scores_pred:
    scores_pred_writer = csv.writer(scores_pred, delimiter=',')

    scores_pred_writer.writerow(['x','id','dataset_id','time','network_structure','epochs','network_error','previously_detected','suspicious_detected','undetected','newly_detected','true_negative_rate','false_negative_rate','false_positive_rate','true_positive_rate','true_positive_rate_timeseries'
])

    with open('scores.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            print (row[4])
            rowDict = ast.literal_eval(row[4])
            print(rowDict['hidden'])
            scores_pred_writer.writerow([row[0],row[1],row[2],row[3],rowDict['hidden'],rowDict['epochs'],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14]])

