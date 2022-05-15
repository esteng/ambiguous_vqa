import copy 
import argparse
from collections import defaultdict
from pathlib  import Path 

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
            data_by_qid[row['Input.question_id']].append(row) 

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



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="/Users/Elias/scratch/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--questions", type=str, default="/Users/Elias/scratch/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    data_dir = "mturk/combined_data"

    questions = read_questions(Path(args.questions))
    annotations = read_annotations(Path(args.annotations))
    qa_lookup = get_qa_lookup(questions, annotations)

    data_by_qid = get_all_data(data_dir) 