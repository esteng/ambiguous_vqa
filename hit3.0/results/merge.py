import argparse
import argparse
import json
import csv 



def merge_csvs(filenames):
    fieldnames = ["HITId","HITTypeId","Title","CreationTime","MaxAssignments","AssignmentDurationInSeconds","AssignmentId","WorkerId","AcceptTime","SubmitTime","WorkTimeInSeconds","Input.answerGroups","Input.answerQuestions","Input.imgUrl","Input.questionStr","Answer.answer_groups","Answer.answer_questions","Answer.is_skip","Answer.skipCheck","Answer.skip_reason","Turkle.Username"]
    data = {filename:[] for filename in filenames}
    for filename in filenames:
        with open(filename) as f1:
            reader = csv.DictReader(f1)
            for i, row in enumerate(reader):
                row['HITId'] = i 
                data[filename].append(row)
    to_ret = []
    for i in range(len(data[filenames[0]])): 
        for filename in filenames:
            row = data[filename][i]
            to_ret.append(row)
    return to_ret 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True, help="path to results csv")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    merged_lines = merge_csvs(args.csvs)
    fieldnames = ["HITId","HITTypeId","Title","CreationTime","MaxAssignments","AssignmentDurationInSeconds","AssignmentId","WorkerId","AcceptTime","SubmitTime","WorkTimeInSeconds","Input.answerGroups","Input.answerQuestions","Input.imgUrl","Input.questionStr","Answer.answer_groups","Answer.answer_questions","Answer.is_skip","Answer.skipCheck","Answer.skip_reason","Turkle.Username"]
    with open(args.out,"w") as f1:
        writer = csv.DictWriter(f1, fieldnames=fieldnames) 
        writer.writeheader()
        for row in merged_lines:
            writer.writerow(row) 
