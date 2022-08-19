import json 
import csv
import argparse
from collections import defaultdict
from tkinter import W

def get_line(line, url_base = "https://cs.jhu.edu/~esteng/images_for_hit/"):
    line_dict = {"imgUrl": '"No link"', # Leave empty
                "questionStr": None, 
                "answerGroups": None, 
                "answerQuestions": None, 
                "question_id": None}    

    
    answer_questions = ['empty' for i in range(len(line["non_repeat_answers"]))] # getting new questions
    questionStr = line['question']
    count = 0
    for answer in line['non_repeat_answers']:
        answer[0]['id'] = 'g.' + str(count) + '.0'
        count += 1
    answerGroups = line['non_repeat_answers']


    # Fill in data
    line_dict['question_id'] = line['question_id'] # question id
    line_dict['questionStr'] = json.dumps(questionStr)
    line_dict['answerGroups'] = json.dumps(answerGroups)
    line_dict['answerQuestions'] = json.dumps(answer_questions)
    return line_dict 

def write_csv(to_write, out_path):
    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=['imgUrl', 'questionStr', 'answerGroups', 'answerQuestions', 'question_id'], lineterminator='\n')
        writer.writeheader()
        for line in to_write:
            writer.writerow(line)

def main(args):
    data = []
    f = open(args.input_json, 'r')
    data = json.load(f)
    to_write = [get_line(data[l]) for l in data]
    write_csv(to_write, args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jason", type=str, dest='input_json', required=True)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True)
    args = parser.parse_args()

    main(args)