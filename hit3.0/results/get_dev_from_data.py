import csv 
from collections import defaultdict
import pdb 
import json 

dev_size = 30
all_data = []
pilot_anns = ["esteng", "A2VIKCIM9TZL22", "A2M03MZWZDXKAJ", "APGX2WZ59OWDN", "A1QUQ0TV9KVD4C", "A2L9763BW12NLA", "ohussei3"]

# first get data from pilot, all of that will go into dev set
with open("mturk/pilot_combined_5_anns.csv") as f1:
    reader = csv.DictReader(f1)
    pilot_data = [row for row in reader]

# Get pilot data inputs to get image urls 
url_lookup = {}
with open("../csvs/mturk/pilot_screening.csv") as f1:
    reader = csv.DictReader(f1)
    for row in reader:
        url_lookup[row['questionStr']] = (row['imgUrl'], row['question_id'])


# collect all unskipped 
pilot_by_qid = defaultdict(list)
for row in pilot_data: 
    if (row["Turkle.Username"] in pilot_anns and 
        row['Answer.is_skip'] == "false"):
        row['Input.imgUrl'], row['Input.question_id'] = url_lookup[row['Input.questionStr']]
        pilot_by_qid[row['Input.question_id']].append(row)

# go through each question id, get one of the annotations if enough people marked as ambiguous 
pilot_done = []
for qid, row_list in pilot_by_qid.items(): 
    # only keep examples for which at least 3 annotators said it was ambiguous 
    if len(row_list) < 3:
        continue
    # go through anns in priority order, with me and the 2 main Mturk workers first 
    for row in row_list:
        for ann in pilot_anns: 
            if row['Input.question_id'] not in pilot_done and row['Turkle.Username'] == ann:
                all_data.append(row)
                pilot_done.append(qid)

# get remaining data from round 1 data 
with open("mturk/full_hit_round_1/A2M03MZWZDXKAJ.csv") as f1:
    reader = csv.DictReader(f1)
    round_1_data = [row for row in reader]

remaining = dev_size - len(all_data)
counter = 0 
for row in round_1_data:
    if row['Answer.is_skip'] == "false" and counter < remaining:
        all_data.append(row)
        counter += 1

# write 
with open("mturk/split/dev_set.csv", "w") as f1:
    writer = csv.DictWriter(f1, fieldnames=all_data[0].keys())
    writer.writeheader()
    writer.writerows(all_data)
