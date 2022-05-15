import copy 
import argparse
from collections import defaultdict
from pathlib  import Path 
import pdb 
import json 

from prep_for_analysis import read_csv, read_annotations, read_questions, get_id_from_url

META_TEMPLATE = {"original_split": "train",
                              "annotation_round": ""}

ANN_TEMPLATE = {"annotator": "",
                "new_clusters": [[""]],
                "new_questions": [""]}

DATA_TEMPLATE = {"question_id": 0,
                 "image_id": 0,
                 "original_question": "",
                 "glove_clusters": [[""]],
                 "multiple_choice_answer": "",
                 "metadata": {},
                 "annotations": [] 
                }

def get_all_data(dir): 
    dir = Path(dir)
    data_by_qid = defaultdict(list)
    for csv in dir.glob("*/*.csv"):
        data = read_csv(csv)
        for row in data: 
            row['annotation_round'] = csv.parent.name
            data_by_qid[int(row['Input.question_id'])].append(row) 

    return data_by_qid

def get_qa_lookup(questions, annotations):
    qa_lookup = {q['question_id']: (q,a) for q,a in zip(questions, annotations)}
    return qa_lookup

def convert_data(data_by_qid, qa_lookup):
    all_data = []
    for qid, data_list in data_by_qid.items():
        question, annotation = qa_lookup[qid]
        jsonl_row = copy.deepcopy(DATA_TEMPLATE)
        metadata = copy.deepcopy(META_TEMPLATE)
        metadata['original_split'] = "train",
        metadata['annotation_round'] = data_list[0]['annotation_round']
        jsonl_row['metadata'] = metadata
        jsonl_row['question_id'] = int(qid)
        jsonl_row['image_id'] = int(get_id_from_url(data_list[0]['Input.imgUrl']))
        jsonl_row['original_question'] = question['question']
        jsonl_row['glove_clusters'] = data_list[0]['Input.answerGroups']
        jsonl_row['multiple_choice_answer'] = annotation['multiple_choice_answer']
        for row in data_list: 
            annotation = copy.deepcopy(ANN_TEMPLATE)
            annotation['annotator'] = row['WorkerId']
            annotation['new_clusters'] = row['Answer.answer_groups']
            annotation['new_questions'] = row['Answer.answer_questions']
            jsonl_row['annotations'].append(annotation)
        all_data.append(jsonl_row) 
    return all_data 

def split_data(all_data, dev_qids): 
    data_by_splits = defaultdict(list)
    dev_qids = [int(x) for x in dev_qids]
    for data_row in all_data:
        if int(data_row['question_id']) in dev_qids: 
            data_by_splits['dev'].append(data_row)
        else:
            data_by_splits['test'].append(data_row)
    return data_by_splits

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/train_anns/annotations.json")
    parser.add_argument("--questions", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/train_anns/questions.json")
    parser.add_argument("--out-dir", type=str, default="json_data/")
    parser.add_argument("--dev_file", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/dev_from_mturk/questions.json")
    args = parser.parse_args()

    data_dir = "mturk_nonskip"

    __, questions = read_questions(Path(args.questions))
    __, annotations = read_annotations(Path(args.annotations))
    qa_lookup = get_qa_lookup(questions, annotations)

    data_by_qid = get_all_data(data_dir) 
    converted = convert_data(data_by_qid, qa_lookup)

    __, dev_questions = read_questions(Path(args.dev_file))
    dev_qids = [x['question_id'].split("_")[0] for x in dev_questions]
    split_data = split_data(converted, dev_qids) 
    for split_name, split_rows in split_data.items(): 
        out_path = Path(args.out_dir).joinpath(f"{split_name}.jsonl")
        with open(out_path, "w") as f1:
            for row in split_rows: 
                f1.write(json.dumps(row) + "\n")