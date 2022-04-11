import json 
import pdb 
import sys 
import pathlib 

full_question_path = sys.argv[1]
full_annotation_path = sys.argv[2]
subset_path = sys.argv[3]
out_path = sys.argv[4]

with open(full_question_path) as f1, open(full_annotation_path) as f2:
    full_question_data = json.load(f1) 
    full_annotation_data = json.load(f2) 

with open(subset_path) as f1: 
    subset_anns = json.load(f1) 


subset_qids = [int(a['qid']) for a in subset_anns]

full_anns = full_annotation_data['annotations'] 
full_questions = full_question_data['questions']
print(f"before") 
print(len(full_anns)) 
print(len(full_questions)) 

full_anns = [a for a in full_anns if a['question_id'] not in subset_qids]
full_questions = [a for a in full_questions if a['question_id'] not in subset_qids]
full_question_data['questions'] = full_questions
full_annotation_data['annotations'] = full_anns 

print(f"after") 
print(len(full_anns)) 
print(len(full_questions)) 

out_path = pathlib.Path(out_path) 
with open(out_path.joinpath("questions.json"), "w") as qf, open(out_path.joinpath("annotations.json"), "w") as af: 
    json.dump(full_question_data, qf) 
    json.dump(full_annotation_data, af) 
