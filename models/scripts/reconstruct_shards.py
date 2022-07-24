import argparse 
import pathlib 
import json 
import re
import pdb

def postprocess(question): 
    question = re.sub("<[^>]*?>", "", question)
    question = question.strip() 
    return question 

def read_pred_file(pred_file):
    predictions_by_qid = {}
    with open(pred_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            qids = line['question_id']
            utts = line['speaker_utterances'][0]
            for qid, question in zip(qids, utts): 
                predictions_by_qid[qid] = postprocess(question)
    return predictions_by_qid

def read_annotation_files(question_file, annotation_file):
    # read in the annotation files
    with open(question_file, 'r') as f:
        questions = json.load(f)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return questions, annotations

def merge(questions, annotations, predictions): 
    missing = []
    answer_list = []
    for i, (q, a) in enumerate(zip(questions, annotations)): 
        answer_list.append(a['answers'][0]['answer'])
        try:
            pred_q = predictions[q['question_id']]
            q['question'] = pred_q
            questions[i] = q
        except KeyError:
            missing.append(q['question_id'])

    return questions, annotations, missing, answer_list


def write_annotation_files(questions, annotations, out_path):
    out_path = pathlib.Path(out_path)
    with open(out_path.joinpath("questions.json"), "w") as qf, open(out_path.joinpath("annotations.json"), "w") as af:
        json.dump(questions, qf, indent=4)
        json.dump(annotations, af, indent=4)   


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, required=True) 
    parser.add_argument("--ann-dir", type=str, default = "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/filtered_line_by_line/")
    parser.add_argument("--forced", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    if args.forced:
        forced = "_forced"
    else:
        forced = "" 
    pred_dir = pathlib.Path(args.pred_dir)
    ann_dir = pathlib.Path(args.ann_dir)
    shards = ann_dir.glob("*")
    # print(list(shards))

    global_questions, global_annotations = [], []
    answer_list = []
    question_header = None
    annotation_header = None
    missing = []
    total = 0

    all_questions_by_qid = {}
    all_annotations_by_qid = {}

    for shard_dir in shards:
        shard = shard_dir.name
        question_file = shard_dir.joinpath("questions.json")
        annotation_file = shard_dir.joinpath("annotations.json")
        question_data, annotation_data = read_annotation_files(question_file, annotation_file)
        all_questions_by_qid.update({x['question_id']: x for x in question_data['questions']})
        all_annotations_by_qid.update({x['question_id']: x for x in annotation_data['annotations']})
        # pdb.set_trace()
        if args.file_name is None:
            pred_file = pred_dir.joinpath(shard + f"_predictions{forced}.jsonl")
        else:
            pred_file = pred_dir.joinpath(args.file_name)
        pred_data = read_pred_file(pred_file)
        new_questions, new_annotations, new_missing, new_answer_list = merge(question_data['questions'], 
                                                                            annotation_data['annotations'], 
                                                                            pred_data) 
        answer_list += new_answer_list
        global_questions += new_questions
        global_annotations += new_annotations
        total += len(new_annotations)
        missing += new_missing
        question_header = question_data
        annotation_header = annotation_data

    question_header['questions'] = global_questions
    annotation_header['annotations'] = global_annotations

    if args.out_dir is None:
        pred_question_path = pred_dir.joinpath(f"{args.split}_predictions{forced}")
    else:
        pred_question_path = pathlib.Path(args.out_dir).joinpath(f"{args.split}_predictions{forced}")
    pred_question_path.mkdir(exist_ok=True)
    write_annotation_files(question_header, annotation_header, pred_question_path) 
    with open(pred_question_path.joinpath("answer_list.json"), "w") as f:
        json.dump(answer_list, f)

    # write missing questions and annotations to file  
    missing_questions, missing_annotations = [], []
    missing_question_header = question_header 
    missing_annotation_header = annotation_header
    for qid in missing:
        missing_questions.append(all_questions_by_qid[qid])
        missing_annotations.append(all_annotations_by_qid[qid])

    missing_path = ann_dir.joinpath("missing")
    missing_path.mkdir(exist_ok=True)
    with open(missing_path.joinpath("questions.json"), "w") as qf, open(missing_path.joinpath("annotations.json"), "w") as af:

        missing_question_header['questions'] = missing_questions
        missing_annotation_header['annotations'] = missing_annotations
        json.dump(missing_question_header, qf, indent=4)
        json.dump(missing_annotation_header, af, indent=4)

        print(f"wrote {len(missing)} missing questions to {missing_path}")
    print(f"{total} total questions")
