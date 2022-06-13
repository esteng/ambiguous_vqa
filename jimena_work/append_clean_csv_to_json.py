import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment
import copy

"""
Sort out non-problematic exampels so we can re-analyze
problematic examples
"""

META_TEMPLATE = {"original_split": "train",
                              "annotation_round": ""}

ANN_TEMPLATE = {"annotator": "",
                "new_clusters": [[""]],
                "new_questions": [""]}

DATA_TEMPLATE = {"question_id": 0,
                 "image_id": 0,
                 "original_question": "",
                 "glove_clusters": [[""]],
                 "multiple_choice_answer": "",
                 "metadata": {},
                 "annotations": [] 
                }

def get_line(line, org_data):
    jsonl_row = copy.deepcopy(DATA_TEMPLATE)
    metadata = copy.deepcopy(META_TEMPLATE)
    metadata['original_split'] = "train" 
    metadata['annotation_round'] = "cleaned_data" 
    jsonl_row['metadata'] = metadata
    jsonl_row['question_id'] = line['Input.question_id']
    jsonl_row['image_id'] = line['Input.imgUrl'] # is url ok or actually image id?
    jsonl_row['original_question'] = line['Input.questionStr']
    jsonl_row['glove_clusters'] = line['Input.answerGroups']
    jsonl_row['multiple_choice_answer'] = line['Input.answerQuestions']  
    
    annotation = copy.deepcopy(ANN_TEMPLATE)
    annotation['annotator'] = org_data[line['HTIId']]  
    annotation['new_clusters'] = line['Answer.answer_groups']
    annotation['new_questions'] = line['Answer.answer_questions']
    jsonl_row['annotations'].append(annotation)

    return jsonl_row 

def write_json(to_write, out_path):
    with open(out_path, "w") as f1:
        for line in to_write:
            f1.write(json.dump(line))

def sort(data):
    org_data = []
    with open(args.input_org_csv) as read_obj_org:
        csv_reader = csv.DictReader(read_obj_org)
        for row in csv_reader:
            org_data[row['HITId']] = row['Turkle.Username']

    delete_count = 0
    flag_count = 0
    delete_flag_count = 0
    sorted_data = []
    for line in data:
        if line["Answer.skip_reason"] == '"flag"' or line["Answer.skip_reason"] == '"delete/flag"':
            flag_count += 1
            continue
        if line["Answer.skip_reason"] == '"delete/flag"':
            delete_flag_count += 1
            continue
        if line["Answer.skip_reason"] == '"delete/flag"' or line["Answer.skip_reason"] == '"delete"':
            delete_count += 1
            continue
        sorted_data.append(line)

    print("Data stats: ")
    print("\tExamples deleted: " + str(delete_count))
    print("\tExamples flagged (kept and deleted): " + str(flag_count))
    exit 
    to_write = [get_line(l, org_data) for l in sorted_data]
    write_json(to_write, args.out_path)

def main(args):
    data = []
    #org_data = []
    with open(args.input_1_csv) as read_obj_1:
        csv_reader = csv.DictReader(read_obj_1)
        for row in csv_reader:
            data.append(row)
    sort(data)
    with open(args.input_2_csv) as read_obj_2:
        csv_reader = csv.DictReader(read_obj_2)
        for row in csv_reader:
            data.append(row)
    sort(data)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-1-csv", type=str, dest='input_1_csv', required=True)
    parser.add_argument("--input-2-csv", type=str, dest='input_2_csv', required=True)
    parser.add_argument("--input-org-csv", type=str, dest='input_org_csv', required=True)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True)
    args = parser.parse_args()

    main(args)