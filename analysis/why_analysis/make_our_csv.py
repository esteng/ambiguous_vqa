import csv 
import pathlib

ann_path = "/home/estengel/annotator_uncertainty/jimena_work/cleaned_data/csv/test_set/consolidate_data_repeat_all_data.csv"
data_path = pathlib.Path("/home/estengel/annotator_uncertainty/hit3.0/csvs/")

data_csvs = data_path.glob("**/*.csv")

all_input_data = []
for csv_file in data_csvs:
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        all_input_data.extend([x for x in reader])

print(all_input_data[0])
input_data_by_qid = {x['question_id']:x  for x in all_input_data if 'question_id' in x.keys()}

with open(ann_path, "r") as f:
    reader = csv.DictReader(f)
    ann_data = [x for x in reader]

ann_data_by_qid = {x['Input.question_id']: x for x in ann_data}

qids_to_keep = []
for qid, row in ann_data_by_qid.items():
    if 'A/P' in row['ambiguity_type'] or 'A/C' in row['ambiguity_type']:
        qids_to_keep.append(qid)
qids_to_keep = set(qids_to_keep)

with open("from_ann.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=all_input_data[0].keys())
    writer.writeheader()
    for qid in qids_to_keep:
        row = input_data_by_qid[qid]
        writer.writerow(row)
