import json 
import pdb 
import sys 
import pathlib 

dev_question_path = sys.argv[1]
dev_annotation_path = sys.argv[2]
dev_len = int(sys.argv[3]) 
out_path = sys.argv[4]

with open(dev_question_path) as f1, open(dev_annotation_path) as f2:
    dev_question_data = json.load(f1) 
    dev_annotation_data = json.load(f2) 


full_anns = dev_annotation_data['annotations'] 
full_questions = dev_question_data['questions']
print(f"before") 
print(len(full_anns)) 
print(len(full_questions)) 

test_question_data = {k:v for k,v in dev_question_data.items() if k != "questions"}
test_annotation_data = {k:v for k,v in dev_question_data.items() if k != "annotations"}

dev_anns = [a for a in full_anns[0:dev_len]]
dev_questions = [a for a in full_questions[0:dev_len]]
dev_question_data['questions'] = dev_questions
dev_annotation_data['annotations'] = dev_anns
print(f"dev: {len(dev_anns)} anns, {len(dev_questions)} questions") 

test_anns = [a for a in full_anns[dev_len:]]
test_questions = [a for a in full_questions[dev_len:]]
test_question_data['questions'] = test_questions
test_annotation_data['annotations'] = test_anns
print(f"test: {len(test_anns)} anns, {len(test_questions)} questions") 

out_path = pathlib.Path(out_path) 
dev_path = out_path.joinpath("dev")
dev_path.mkdir(exist_ok=True, parents=True)
with open(dev_path.joinpath("questions.json"), "w") as qf, open(dev_path.joinpath("annotations.json"), "w") as af: 
    json.dump(dev_question_data, qf) 
    json.dump(dev_annotation_data, af) 

test_path = out_path.joinpath("test")
test_path.mkdir(exist_ok=True, parents=True)
with open(test_path.joinpath("questions.json"), "w") as qf, open(test_path.joinpath("annotations.json"), "w") as af: 
    json.dump(test_question_data, qf) 
    json.dump(test_annotation_data, af) 