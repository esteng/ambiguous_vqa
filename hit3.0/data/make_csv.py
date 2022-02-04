from email.policy import default
import json
import csv 
import argparse 
from collections import defaultdict

def get_answer_groups(answers, group_ids):
    groups = defaultdict(set)
    for a, id in zip(answers, group_ids): 
        groups[id].update({a})
    just_answers = [list(x) for x in groups.values()]
    to_ret = []
    for i, group in enumerate(just_answers): 
        to_group = []
        for j, ans in enumerate(group):
            to_group.append({"id": f"g{i}.{j}", "content": ans})
        to_ret.append(to_group)
    return to_ret

def get_line(line, url_base = "https://cs.jhu.edu/~esteng/images_for_hit/"):
    line_dict = {"imgUrl": None,
                "questionStr": None,
                "answerGroups": None, 
                "answerQuestions": None}    

    image_url = f"{url_base}/{line['image']}"
    question_str = line['question']
    answer_groups = get_answer_groups(line['answers'], line['ans_groups'])
    answer_questions = [question_str for i in range(len(answer_groups))]

    line_dict['imgUrl'] = json.dumps(image_url) 
    line_dict['questionStr'] = json.dumps(question_str) 
    line_dict['answerGroups'] = json.dumps(answer_groups)
    line_dict['answerQuestions'] = json.dumps(answer_questions)
    return line_dict 
    
def write_csv(to_write, out_path): 
    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=['imgUrl', 'questionStr', 'answerGroups', 'answerQuestions'])
        writer.writeheader()
        for line in to_write:
            writer.writerow(line)
        

def main(args): 
    with open(args.input_json) as f1:
        data = json.load(f1)
    to_write = [get_line(l) for l in data]
    write_csv(to_write, args.out_path) 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True, help="input csv that has the cluster information, url information, and questions/answers")
    parser.add_argument("--out-path", type=str, required=True, help = "path to the output csv file")
    args = parser.parse_args() 

    main(args) 