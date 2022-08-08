import json
import csv 
import pickle as pkl 


with open("missing.pkl", "rb") as f:
    missing = pkl.load(f)


annotations = json.load(open("/home/estengel/annotator_uncertainty/jimena_work/cleaned_data/csv/test_set/annotations.json"))
questions = json.load(open("/home/estengel/annotator_uncertainty/jimena_work/cleaned_data/csv/test_set/questions.json"))

missing_questions, missing_annotations = [], []
for q, a in zip(questions['questions'], annotations['annotations']):
    if q['question_id'] in missing:
        missing_questions.append(q)
        missing_annotations.append(a)

questions['questions'] = missing_questions
annotations['annotations'] = missing_annotations

with open("/home/estengel/annotator_uncertainty/jimena_work/cleaned_data/csv/missing_test_set/annotations.json", "w") as annf,\
    open("/home/estengel/annotator_uncertainty/jimena_work/cleaned_data/csv/missing_test_set/questions.json", "w") as qf:
    json.dump(annotations, annf)
    json.dump(questions, qf)