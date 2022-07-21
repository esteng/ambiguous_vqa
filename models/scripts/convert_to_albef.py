# script to convert filtered data to ablef format 
import json 
import pdb
import re 
import sys 
import pathlib 
import argparse 
import copy


def get_image_id(iid):
    correct_len = len("000000262148") 
    iid = str(iid)
    iid_len = len(iid)
    n_zeros = correct_len - iid_len
    zeros = "0" * n_zeros
    iid = zeros + iid
    path = "train2014/COCO_train2014_" + iid + ".jpg"
    return path 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/filtered_line_by_line/")
    parser.add_argument("--out-dir", default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/albef/data/")
    parser.add_argument("--filename", default="vqa_train_filtered_line_by_line.json") 
    args = parser.parse_args()
    
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    with open(data_dir.joinpath("questions.json"), "r") as qf:
        questions = json.load(qf)['questions']
    with open(data_dir.joinpath("annotations.json"), "r") as af:
        annotations = json.load(af)['annotations']

    data = []    
    for q, a in zip(questions,annotations): 

        if a['answers'][0]['answer_confidence'] == "no":
            continue

        image_id = get_image_id(q['image_id'])
        instance = {"question": q['question'],
                    "question_id": q['question_id'], 
                    "answer": [a['answers'][0]['answer']],
                    "image": image_id, 
                    "dataset": "vqa"}

        data.append(instance)

    with open(out_dir.joinpath(args.filename), "w") as f:
        json.dump(data, f, indent=4)

