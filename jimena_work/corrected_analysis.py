import argparse
import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
import pdb
from scipy.optimize import linear_sum_assignment

def write_csv(input_csv, out_path):
    with open(input_csv) as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row['']

def main(args):
    write_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, dest='input_csv', required=True)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True)
    args = parser.parse_args()

    main(args)