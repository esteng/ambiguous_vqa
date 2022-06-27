import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment
import copy

def search_count(data, args):
    search_item = args.key
    match_count = 0

    for row in data:
        temp = row['Answer.skip_reason'].strip('"')
        labels = temp.split('.')

        if search_item in labels:
            #print('Hello')
            print(row['Answer.skip_reason'])
            match_count += 1
    
    print(match_count)

    
            
    

def main(args):
    data = []
    with open(args.input_csv) as read_obj:
        csv_reader = csv.DictReader(read_obj)
        for row in csv_reader:
            data.append(row)
    
    search_count(data, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, dest='input_csv', required=True)
    parser.add_argument("--search-key", type=str, dest='key', required=False)
    args = parser.parse_args()

    main(args)