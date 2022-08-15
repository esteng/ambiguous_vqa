import csv 
import pathlib 

top_dir = pathlib.Path("/home/estengel/annotator_uncertainty/hit3.0/results/mturk") 

all_turkers = []
for file in top_dir.glob("*/*.csv"):
    with open(file) as f1:
        reader = csv.DictReader(f1) 
        for row in reader:
            all_turkers.append(row['WorkerId']) 

all_turkers = set(all_turkers) 
print(all_turkers)

datapoint = {"Worker ID": None, "UPDATE-worked_on_ambiguity": 100}
to_write = []
with open("worked_on_annotation.csv", "w") as f1:
    writer = csv.DictWriter(f1, fieldnames=datapoint.keys()) 
    writer.writeheader() 
    for worker in all_turkers:
        datapoint["Worker ID"] = worker 
        writer.writerow(datapoint)




