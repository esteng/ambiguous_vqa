import csv 
import pathlib
import json 
import argparse

def read_csv(path):
    all_data = []
    with open(path) as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            data = {}
            for k, v in row.items():
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    v = v
                data[k] = v
            all_data.append(data)
    return all_data

def annotate(data, resume_path): 
    if resume_path.exists():
        with open(resume_path) as f1:
            data = json.load(f1)
    for i, row in enumerate(data):
        if row['Answer.is_skip']:
            continue
        if 'is_dynamic' in row.keys() and row['is_dynamic'] is not None:
            # already annotated 
            continue
        print(row['Input.questionStr'])
        is_dynamic = input("\tDynamic? ")
        is_agentive = input("\tAgentive? ")
        row['is_dynamic'] = is_dynamic
        row['is_agentive'] = is_agentive
        data[i] = row
        save_annotations(data, resume_path)

def save_annotations(data, path):
    with open(str(path), "w") as f1:
        json.dump(data, f1, indent=4)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="output.csv")
    parser.add_argument("--out-file", type=str, default = "dynamic_agentive.json")
    args = parser.parse_args()

    data = read_csv(args.csv)
    out_file = pathlib.Path(args.out_file)
    annotate(data, out_file)