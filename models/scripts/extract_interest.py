import json 
import pdb 
import sys 
import re
import pathlib 
import argparse

def is_kind_question(question):
    if question[0:2] == ["what", "kind"] or question[0:2] == ['what', 'type']:
        return True
    return False

def is_location_question(question):
    if question[0].strip() == "where" or question[-1].strip() == "where":
        return True
    joined_question = " ".join(question)
    if "what is the location" in joined_question:
        return True
    if "what 's the location" in joined_question:
        return True
    if "what's the location" in joined_question:
        return True
    return False

def is_why_question(question): 
    if question[0].strip() == "why":
        return True
    return False

def is_interest(question): 
    question = re.split("\s+", question)
    question = [x.lower() for x in question]
    return is_kind_question(question) or is_location_question(question) or is_why_question(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--prediction_path", type=str, default=None)
    args = parser.parse_args()
    ann_path = pathlib.Path(args.ann_path)
    out_path = pathlib.Path(args.out_path)
    question_path = ann_path.joinpath("questions.json")
    annotation_path = ann_path.joinpath("annotations.json")
    with open(question_path) as f1, open(annotation_path) as f2:
        full_question_data = json.load(f1) 
        full_annotation_data = json.load(f2) 

    questions = full_question_data['questions']
    annotations = full_annotation_data['annotations']
    n_interest = 0
    total = 0 
    new_qs, new_as = [], []
    for q, a in zip(questions, annotations): 
        question = q['question']
        if is_interest(question):
            new_qs.append(q)
            new_as.append(a) 
            n_interest += 1
        total += 1

    new_q_data = full_question_data
    new_q_data['questions'] = new_qs
    new_a_data = full_annotation_data
    new_a_data['annotations'] = new_as

    print(f"perc of interest: {n_interest/total*100:.2f}")

    with open(out_path.joinpath("questions.json"), "w") as qf, open(out_path.joinpath("annotations.json"), "w") as af:
        json.dump(new_q_data, qf)
        json.dump(new_a_data, af)

    new_q_ids = [x['question_id'] for x in new_qs]

    prediction_subset = []
    if args.prediction_path is not None:
        data = json.load(open(args.prediction_path))
        for line in data:
            if line['question_id'] in new_q_ids:
                prediction_subset.append(line) 

    pred_path = pathlib.Path(args.prediction_path).parent

    with open(pred_path.joinpath("interest_subset.json"), "w") as f:
        json.dump(prediction_subset, f)