import argparse
import argparse
import json
import csv 


def process_row(row):
    columns_to_json = ['Answer.answer_groups', "Answer.answer_questions", "Input.answerGroups", "Input.answerQuestions"]
    for col in columns_to_json:
        # print(row[col])
        row[col] = json.loads(row[col])
    did_skip = True if row['Answer.is_skip'] == "true" else False
    row['Answer.is_skip'] = did_skip
    return row

def process_csv(filename):
    fieldnames = ["HITId","HITTypeId","Title","CreationTime","MaxAssignments","AssignmentDurationInSeconds","AssignmentId","WorkerId","AcceptTime","SubmitTime","WorkTimeInSeconds","Input.answerGroups","Input.answerQuestions","Input.imgUrl","Input.questionStr","Answer.answer_groups","Answer.answer_questions","Answer.is_skip","Answer.skipCheck","Answer.skip_reason","Turkle.Username"]
    to_ret = []
    with open(filename) as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            to_ret.append(process_row(row)) 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to results csv")
    args = parser.parse_args()

    process_csv(args.csv)