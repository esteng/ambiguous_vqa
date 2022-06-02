import json 
import csv
import argparse
from collections import defaultdict

def get_answer_groups(annotations):
    data = annotations[0] # get annotations dict
    new_clusters = data["new_clusters"]
    groups = defaultdict(set)

    to_ret = []
    i = 0
    for cluster in new_clusters:
        j = 0
        for object in cluster:
            ans = object["content"]
            to_group = []
            to_group.append({"id": f"g{i}.{j}", "content": ans})
            j += 1
        to_ret.append(to_group)
        i += 1
    return to_ret

'''
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
'''

def get_line(line, url_base = "https://cs.jhu.edu/~esteng/images_for_hit/"):
    line_dict = {"imgUrl": None,
                "questionStr": None, 
                "answerGroups": None, # From annotator
                "answerQuestions": None, # From annotator
                "question_id": None}    

    image_url = f"{url_base}{line['image_id']}"
    question_str = line['original_question']
    # To do: 
    answer_groups = get_answer_groups(line['annotations']) # getting new groups
    annotator_1 = line['annotations'][0]
    answer_questions = [annotator_1['new_questions'][i] for i in range(len(answer_groups))] # getting new questions

    # metadata
    line_dict['question_id'] = line['question_id'] # question id
    line_dict['imgUrl'] = json.dumps(image_url) # image url
    line_dict['questionStr'] = json.dumps(question_str) # question string
    # To do: 
    line_dict['answerGroups'] = json.dumps(answer_groups) # annotator answer groups
    line_dict['answerQuestions'] = json.dumps(answer_questions) # annotator group questions
    return line_dict 

def write_csv(to_write, out_path):
    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=['imgUrl', 'questionStr', 'answerGroups', 'answerQuestions', 'question_id'])
        writer.writeheader()
        for line in to_write:
            writer.writerow(line)

def main(args):
    data = []
    for line in open(args.input_json, 'r'):
        data.append(json.loads(line))
    to_write = [get_line(l) for l in data]
    write_csv(to_write, args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jason", type=str, dest='input_json', required=True)
    parser.add_argument("--out-path", type=str, dest='out_path', required=True)
    args = parser.parse_args()

    main(args)