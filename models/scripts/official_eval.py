import argparse
import pathlib 
import json 
import pdb 
import subprocess
from tqdm import tqdm 
import copy 

from transformers import ViltModel
from eval.vqa import VQA
from eval.vqa_eval import VQAEval

def get_lookup(model_name_or_path): 
    model = ViltModel.from_pretrained(model_name_or_path)
    lookup = copy.deepcopy(model.config.label2id)
    del(model)
    lookup = {v:k for k,v in lookup.items()}
    return lookup 

def get_predictions_from_file(predictions_file, lookup):

    predictions = []
    # get line total of predictions file with wc -l
    line_total = int(subprocess.check_output(['wc', '-l', predictions_file]).split()[0])

    with open(predictions_file, 'r') as f:
        qidx = 0
        for batch in tqdm(f, total = line_total): 
            batch = json.loads(batch)
            answers = batch['predicted_labels']
            qids = batch['question_id']
            for i, ans in enumerate(answers):
                qid = qids[i]
                try:
                    if lookup is not None:
                        answer_string = lookup[ans]
                    else:
                        answer_string = ans
                except KeyError:
                    answer_string = "ERROR"
                prediction = {"question_id": qid, "answer": answer_string}
                predictions.append(prediction)
                qidx += 1

    return predictions 

def get_predictions_albef(path):
    with open(path) as f1:
        data = json.load(f1)
    
    return data 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--reference", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val_anns/")
    parser.add_argument("--model-name", type=str, default="/brtx/605-nvme1/estengel/annotator_uncertainty/models/finetune_vilt_pytorch/")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--from-albef", action="store_true")
    args = parser.parse_args()

    reference = pathlib.Path(args.reference)
    annotation_file = str(reference.joinpath("annotations.json")) 
    question_file = str(reference.joinpath("questions.json"))
    pred_path = pathlib.Path(args.predictions).parent
    if args.model_name is not None:
        lookup = get_lookup(args.model_name)
    else:
        lookup = None

    if (args.recompute and pred_path.joinpath("pred.json").exists()) or not pred_path.joinpath("pred.json").exists():
        if args.from_albef:
            predictions = get_predictions_albef(args.predictions) 
            with open(pred_path.joinpath("pred.json"), "w") as f1:
                json.dump(predictions, f1)
        else:
            predictions = get_predictions_from_file(args.predictions, lookup)
            with open(pred_path.joinpath("pred.json"), "w") as f1:
                json.dump(predictions, f1)
    else:
        if args.from_albef:
            predictions = get_predictions_albef(args.predictions) 
            with open(pred_path.joinpath("pred.json"), "w") as f1:
                json.dump(predictions, f1)
        else:
            predictions = json.load(open(pred_path.joinpath("pred.json")))
    
            with open(pred_path.joinpath("pred.json"), "w") as f1:
                json.dump(predictions, f1)
    vqa = VQA(annotation_file=annotation_file, question_file=question_file)

    # see if there are missing ones 
    pred_qids= [q['question_id'] for q in predictions]
    ref_qids = vqa.getQuesIds()
    missing = set(ref_qids) - set(pred_qids)
    if len(missing) > 0:
        for qid in missing:
            predictions.append({"question_id": qid, "answer": "ERROR"})
        with open(pred_path.joinpath("pred.json"), "w") as f1:
            json.dump(predictions, f1)

    vqa_res = vqa.loadRes(pred_path.joinpath("pred.json"), question_file)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)
    vqa_eval.evaluate()
    print(f"Overall Accuracy: {vqa_eval.accuracy['overall']}")
