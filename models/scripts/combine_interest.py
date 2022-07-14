import json 
import pdb 
import sys 
import re
import pathlib 
import argparse
from official_eval import get_predictions_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", type=str)
    parser.add_argument("--pred-path", type=str)
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()
    ann_path = pathlib.Path(args.ann_path)
    out_path = pathlib.Path(args.out_path)
    question_path = ann_path.joinpath("questions.json")
    annotation_path = ann_path.joinpath("annotations.json")
    with open(question_path) as f1, open(annotation_path) as f2:
        full_question_data = json.load(f1) 
        full_annotation_data = json.load(f2) 


    predictions = get_predictions_from_file(args.pred_path, None) 
    pred_by_qid = {x['question_id']: x for x in predictions}

    questions = full_question_data['questions']
    annotations = full_annotation_data['annotations']
    n_interest = 0
    total = 0 
    new_qs, new_as = [], []
    for i, (q, a) in enumerate(zip(questions, annotations)): 
        if q['question_id'] in pred_by_qid.keys():
            q['question'] = pred_by_qid[q['question_id']]['answer']
            questions[i] = q 

    new_q_data = full_question_data
    new_q_data['questions'] = questions
    new_a_data = full_annotation_data
    new_a_data['annotations'] = annotations

    print(f"perc of interest: {n_interest/total*100:.2f}")

    with open(out_path.joinpath("questions.json"), "w") as qf, open(out_path.joinpath("annotations.json"), "w") as af:
        json.dump(new_q_data, qf)
        json.dump(new_a_data, af)
gg