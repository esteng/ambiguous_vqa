import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment
import copy
import os

from sqlalchemy import desc

def main(args):
    directory = os.fsencode(args.dir)

    data = {}

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open('./mturk_nonskip/' + filename) as read_obj:
            csv_reader = csv.DictReader(read_obj)
            for row in csv_reader:
                if row['Input.question_id'] not in data:
                    data[row['Input.question_id']] = []
                    data[row['Input.question_id']].append(row['Answer.answer_groups'])
                else:
                    data[row['Input.question_id']].append(row['Answer.answer_groups'])

    with open(args.out, "w") as f1:
        json_string = json.dumps(data, indent=4)
        f1.write(json_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, dest='dir', required=True, help='directory of files')
    parser.add_argument("--out-path", type=str, dest='out', required=True, help='output file')

    args= parser.parse_args()

    main(args)