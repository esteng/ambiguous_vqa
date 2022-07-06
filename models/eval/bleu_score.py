import argparse
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import numpy as np 

from util import read_test_data, read_generations, match_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    args = parser.parse_args()

    questions, annotations = read_test_data(Path(args.ann_path))
    pred_data = read_generations(Path(args.pred_path))

    pairs = match_data(questions, annotations, pred_data)

    all_bleu_scores = []

    for gold, pred in pairs: 
        bleu_tuple = []
        weights = [(1, 0, 0, 0),
                  (1./2., 1./2., 0, 0),
                  (1./3., 1./3., 1./3., 0),
                  (1./4., 1./4., 1./4., 1./4.)]
        bleu_scores = sentence_bleu([gold], pred, weights = weights)
        # print(f"gold: {gold}")
        # print(f"pred: {pred}")
        # print(f"scores: {bleu_scores}")
        all_bleu_scores.append(bleu_scores)

    all_bleu_scores = np.array(all_bleu_scores)
    mean_bleu_score = np.mean(all_bleu_scores, axis=0) 
    print(f"prediction file: {args.pred_path}")
    for i in range(4):
        print(f"BLEU-{i}: {mean_bleu_score[i]:.2f}")