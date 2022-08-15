import csv 
import pathlib 


all_turkers = ['A1K8VUKRL53OX']


datapoint = {"Worker ID": None, "UPDATE-IsA1K8VUKRL53OX": 100}
to_write = []
with open("IsA1K8VUKRL53OX.csv", "w") as f1:
    writer = csv.DictWriter(f1, fieldnames=datapoint.keys()) 
    writer.writeheader() 
    for worker in all_turkers:
        datapoint["Worker ID"] = worker 
        writer.writerow(datapoint)




