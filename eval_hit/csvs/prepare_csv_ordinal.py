import csv 
import pdb 
import numpy as np 

np.random.seed(12) 

# read data csv 
data = []
with open("../../jimena_work/cleaned_data/csv/consolidate_data_repeat_all_data.csv") as f1:
    reader = csv.DictReader(f1)
    data = [x for x in reader]

# sample 100 
sample = np.random.choice(data, size=100, replace=False)



# 