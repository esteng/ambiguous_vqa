import argparse
import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment

"""
Sort out nom problematic exampels so we can re analyze
problem examples
"""

def get_line(line):
    line_dict = {"imgUrl": None,
                "questionStr": None, 
                "answerGroups": None, # From annotator
                "answerQuestions": None, # From annotator
                "question_id": None}

    line_dict['question_id'] = line['Input.question_id'] # question id
    line_dict['imgUrl'] = line['Input.imgUrl'] # image url
    line_dict['questionStr'] = line['Input.questionStr'] # question string
    # To do: 
    line_dict['answerGroups'] = line['Answer.answer_groups'] # annotator answer groups
    line_dict['answerQuestions'] = line['Answer.answer_questions'] # annotator group questions
    return line_dict 

def write_csv(to_write, out_path):
    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=['imgUrl', 'questionStr', 'answerGroups', 'answerQuestions', 'question_id'])
        writer.writeheader()
        for line in to_write:
            writer.writerow(line)

def sort(data):
    sorted_data = []
    for line in data:
        if line["Answer.skip_reason"] == '"flag"' or line["Answer.skip_reason"] == '"flag"':
            sorted_data.append(line)
    to_write = [get_line(l) for l in sorted_data]
    write_csv(to_write, args.out_path)

def main(args):
    data = []
    with open(args.input_csv) as read_obj:
        csv_reader = csv.DictReader(read_obj)
        for row in csv_reader:
            data.append(row)
    sort(data)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, dest='input_csv', required=True)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True)
    args = parser.parse_args()

    main(args)