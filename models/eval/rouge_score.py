import argparse
from pathlib import Path
import pdb 

import numpy as np 
from rouge import Rouge

from util import read_test_data, read_generations, match_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    args = parser.parse_args()

    questions, annotations = read_test_data(Path(args.ann_path))
    pred_data = read_generations(Path(args.pred_path))

    pairs = match_data(questions, annotations, pred_data, tokenize=False)

    all_rouge_scores = []
    rouge = Rouge()
    golds, preds = [], []
    for gold, pred in pairs:
        golds.append(gold)
        preds.append(pred)
    score_dict = rouge.get_scores(preds, golds, avg=True)

    avg_score = np.mean(score_dict['rouge-l']['f'])
    print(f"prediction file: {args.pred_path}")
    print(f"ROUGE-L: {avg_score:.2f}")