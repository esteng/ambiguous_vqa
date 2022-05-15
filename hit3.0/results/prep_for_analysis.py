import csv 
from pathlib import Path 
import json 
import argparse
import copy
import pdb 
import re


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)

        rows = [row for row in reader]
        for i, row in enumerate(rows):
            for k,v in row.items():
                try:
                    v = json.loads(v)
                except:
                    pass
                row[k] = v
            rows[i] = row
    return rows 

def read_annotations(path):
    data = json.load(open(path))
    return data, data['annotations']

def read_questions(path):
    data = json.load(open(path))
    return data, data['questions'] 

def get_id_from_url(url):
    print(url)
    filename = url.split("/")[-1].split(".")[0]
    # get last part of filename 
    filename = filename.split("_")[-1]
    # remove non-digits 
    filename = re.sub(r"[^\d]", "", filename)
    # delete leading zeros 
    fname_to_keep = int(re.sub("^0+","", filename))
    return fname_to_keep

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="mturk/full_hit_round_1") 
    parser.add_argument("--annotations", type=str, default="/Users/Elias/scratch/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--questions", type=str, default="/Users/Elias/scratch/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--out-dir", type=str, default="/Users/Elias/scratch/qa_from_mturk/")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args() 

    results_dir = Path(args.results_dir)
    print("reading csvs...")
    csvs = [read_csv(x) for x in results_dir.glob("*.csv")]
    print("reading anns...")
    annotations_data, annotations= read_annotations(Path(args.annotations))
    print("reading questions...")
    questions_data, questions = read_questions(Path(args.questions))

    examples = []

    qa_lookup = {q['question_id']: (q,a) for q,a in zip(questions, annotations)}

    for csv in csvs: 
        for row in csv:
            if row['Answer.is_skip'] in ["False", "false", False]:
                answer_groups = row['Answer.answer_groups']
                answer_questions = row['Answer.answer_questions']
                image_id = get_id_from_url(row['Input.imgUrl']) 
                original_question = row['Input.questionStr']
                question_id = row['Input.question_id']
                question_template, annotation_template = qa_lookup[question_id]
                counter = 0 
                for i in range(len(answer_groups)):
                    question = answer_questions[i]
                    group = answer_groups[i]
                    done_answers = []
                    for j, answer in enumerate(group): 
                        if answer['content'] not in done_answers: 
                            q = copy.deepcopy(question_template)
                            a = copy.deepcopy(annotation_template)
                            q['image_id'] = int(image_id)
                            q['question'] = original_question
                            q['new_question'] = question 
                            q['question_id'] = f"{question_id}_{counter}"
                            a['image_id'] = int(image_id)
                            a['question_id'] = f"{question_id}_{counter}" 
                            a['multiple_choice_answer'] = answer['content']
                            a['answers'] = [{'answer_confidence': 'yes',
                                            'answer': answer['content'], 
                                            'answer_id': counter,
                                            'mturk_id': answer['id']}]
                            examples.append((q,a))
                            done_answers.append(answer['content']) 
                            counter += 1

    
    new_questions_data = questions_data
    new_annotations_data = annotations_data

    if args.limit is not None:
        examples = examples[:args.limit]

    print(f"There are {len(examples)} unique image-question-answer triples") 

    new_questions_data['questions'] = [x[0] for x in examples]
    new_annotations_data['annotations'] = [x[1] for x in examples]

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    
    with open(out_path.joinpath("questions.json"),"w") as qf,\
        open(out_path.joinpath("annotations.json"),"w") as af: 
        json.dump(new_questions_data, qf, indent=4)
        json.dump(new_annotations_data, af, indent=4)











