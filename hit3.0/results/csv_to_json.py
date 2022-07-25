from ast import arg
import json
import csv 
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/train_anns/annotations.json")
    parser.add_argument("--questions", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/train_anns/questions.json")
    parser.add_argument("--dev-csv", type=str, default = "mturk/split/dev_set.csv")
    parser.add_argument("--out-dir", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/dev_from_mturk/")
    args = parser.parse_args()

    with open(args.questions) as f1:
        question_data = json.load(f1)
    with open(args.annotations) as f1:
        annotation_data = json.load(f1)

    with open(args.dev_csv) as f1:
        reader = csv.DictReader(f1)
        qids = []
        for line in reader: 
            qids.append(line['Input.question_id'])

    question_data['questions'] = [x for x in question_data['questions'] if x['question_id'] in qids]
    annotation_data['annotations'] = [x for x in annotation_data['annotations'] if x['question_id'] in qids]
    # write questions and annotations  
    with open(args.out_dir + "questions.json", "w") as f1:
        json.dump(question_data, f1)

    with open(args.out_dir + "annotations.json", "w") as f1:
        json.dump(annotation_data, f1)