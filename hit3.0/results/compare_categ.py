import csv 
import re 
import argparse 
import json
import pdb 

def preprocess(answer_str): 
    answer_str = json.loads(answer_str)
    # remove parentheticals 
    answer_str = re.sub(r'\([^)]*\)', '', answer_str)
    answer_lst = re.split("[.,]", answer_str)
    answer_lst = [x.strip() for x in answer_lst]
    answer_lst = [x for x in answer_lst if len(x) > 0]
    return set(answer_lst)

def read_csv(path):
    # read csvs
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def compare(csv1, csv2): 
    csv1_data = read_csv(csv1)
    csv2_data = read_csv(csv2)
    # compare
    agree = 0
    total = 0
    disagree_rows = []
    for row1, row2 in zip(csv1_data, csv2_data): 
        ans1 = preprocess(row1['Answer.skip_reason'])
        ans2 = preprocess(row2['Answer.skip_reason'])
        if ans1 == ans2: 
            agree += 1
        else:
            disagree_rows.append((row1, row2))
        total += 1
    print(f"agreed on {agree/total*100:.2f}% of cases ({agree}/{total})")
    return disagree_rows

def examine(disagree_rows):
    for r1, r2 in disagree_rows:
        print(f"ann: {r1['Turkle.Username']} said: {preprocess(r1['Answer.skip_reason'])}")
        print(f"ann: {r2['Turkle.Username']} said: {preprocess(r2['Answer.skip_reason'])}")
        pdb.set_trace() 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("csv1", help="path to csv1")
    parser.add_argument("csv2", help="path to csv2")
    args = parser.parse_args()
    disagree_rows = compare(args.csv1, args.csv2)
    examine(disagree_rows)