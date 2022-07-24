import argparse
from pathlib import Path
import pdb 

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
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

    weights = [(1, 0, 0, 0),
                (1./2., 1./2., 0, 0),
                (1./3., 1./3., 1./3., 0),
                (1./4., 1./4., 1./4., 1./4.)]
    refs = []
    preds = []
    for gold, pred in pairs: 
        bleu_tuple = []
        refs.append([gold])
        preds.append(pred)
    mean_bleu_score = corpus_bleu(refs, preds, weights)
    print(f"prediction file: {args.pred_path}")
    for i in range(4):
        print(f"BLEU-{i+1}: {mean_bleu_score[i]:.2f}")