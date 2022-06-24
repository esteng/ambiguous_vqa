import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment
import copy

def sort(data):
    

def main(args):
    data = []
    with open(args.input_csv) as read_obj:
        csv_reader = csv.DictReader(read_obj)
        for row in csv_reader:
            data.append(row)
    
    sort(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inut-csv", type=str, dest='input_csv', required=True)

    main(args)