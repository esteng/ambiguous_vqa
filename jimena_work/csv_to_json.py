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
appended csv -> json
"""

META_TEMPLATE = {"original_split": "train",
                              "annotation_round": ""}

ANN_TEMPLATE = {"annotator": "",
                "new_clusters": [[""]],
                "new_questions": [""],}

DATA_TEMPLATE = {"question_id": 0,
                 "image_id": 0,
                 "original_question": "",
                 "glove_clusters": [[""]],
                 "multiple_choice_answer": "",
                 "metadata": {},
                 "annotations": [], 
                 "ambiguity_type": ""
                }

def get_line(line, amb_dict = {}, username_dict = {}):
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
    jsonl_row['ambiguity_type'] = amb_dict.get(line['Input.question_id'])
    
    
    annotators = username_dict.get(line['Input.question_id'])

    if annotators != None:
        for dic in annotators:
            annotation = copy.deepcopy(ANN_TEMPLATE)
            annotation['annotator'] = dic['Username']  
            annotation['new_clusters'] = dic['Answer_groups']
            annotation['new_questions'] = line['Answer.answer_questions']
            jsonl_row['annotations'].append(annotation)

    return jsonl_row 

def write_json(to_write, out_path):
    with open(out_path, "w") as f1:
        for line in to_write:
            json.dump(line, f1)

def sort(data, amb_dict = {}, username_dict = {}):

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
        '''
        if args.include == 'include': 
            amb_list = amb_dict.get([line['Input.question_id']])
            if amb_list.len() == 1 and amb_list[0].strip('"').strip('\\') == 'U':
                continue
            if 'M/A' in amb_list or 'M/B' in amb_list:
                continue
        '''
        sorted_data.append(line)

    print("Data stats: ")
    print("\tExamples deleted: " + str(delete_count))
    print("\tExamples flagged (kept and deleted): " + str(flag_count))
    exit 
    to_write = [get_line(l, amb_dict, username_dict) for l in sorted_data]
    write_json(to_write, args.out_path)

def main(args):
    data = []
    
    with open(args.csv_1) as read_obj_1:
        csv_reader = csv.DictReader(read_obj_1)
        for row in csv_reader:
            data.append(row)
    sort(data)
    # If we only want to append some csv data
    if args.csv_2 != None and args.amb_csv == None:
        with open(args.csv_2) as read_obj_2:
            csv_reader = csv.DictReader(read_obj_2)
            for row in csv_reader:
                data.append(row)
        sort(data)
    # If we want to append csv data and consolidate ambugity data
    elif args.csv_2 != None and args.amb_csv != None:
        by_question_id_ambguity_dict = {}
        by_hitid_annotation_dict = {}
        with open(args.amb_csv) as read_obj_2:
            csv_reader = csv.DictReader(read_obj_2)
            for row in csv_reader:
                ambiguity_list = row['Answer.skip_reason'].strip('[]"').split(',')
                by_question_id_ambguity_dict[row['Input.question_id']] = ambiguity_list[0]
        with open(args.input_org_csv) as read_obj_3:
            csv_reader = csv.DictReader(read_obj_3)
            for row in csv_reader:
                if row['question_id'] not in by_hitid_annotation_dict:
                    by_hitid_annotation_dict[row['question_id']] = []
                    temp_dict = {}
                    temp_dict['Username'] = 111
                    temp_dict['Answer_groups'] = row['answerGroups']
                    by_hitid_annotation_dict[row['question_id']].append(temp_dict)
                else:
                    temp_dict = {}
                    temp_dict['Username'] = 1111
                    temp_dict['Answer_groups'] = row['answerGroups']
                    by_hitid_annotation_dict[row['question_id']].append(temp_dict)
            print(by_hitid_annotation_dict)
                
        #print(by_question_id_turkle_username_dict)
        
        with open(args.csv_2) as read_obj_2:
            csv_reader = csv.DictReader(read_obj_2)
            for row in csv_reader:
                data.append(row)
        sort(data, by_question_id_ambguity_dict, by_hitid_annotation_dict)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-inputs-csv", type=str, dest='csv_1', required=True, help='csv data') # Clean data
    parser.add_argument("--append-csv", type=str, dest='csv_2', required=False, help='csv data to be appended') # Corrected clean data 
    parser.add_argument("--ambiguity-data-csv", type=str, dest='amb_csv', required=False, help='ambiguity data to be consolidated') # Ambiguity data
    parser.add_argument("--input-org-csv", type=str, dest='input_org_csv', required=False) # Original annotator data (with annotator usernames)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True, help='out path') # Output file

    parser.add_argument("--include", type=str, dest='include', required=True)
    args = parser.parse_args()

    main(args)