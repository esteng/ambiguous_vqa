import json 
import sys

path = sys.argv[1]

with open(path) as f1:
    data = json.load(f1) 

data['dataset_reader']['add_force_word_ids'] = True
data['validation_dataset_reader']['add_force_word_ids'] = True
data['data_loader']['drop_last'] = False

with open(path, "w") as f1:
    json.dump(data, f1, indent=4) 
