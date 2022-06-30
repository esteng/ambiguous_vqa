import argparse
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
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

    all_meteor_scores = []

    for gold, pred in pairs: 
        meteor_scores = meteor_score([gold], pred)
        all_meteor_scores.append(meteor_scores)

    mean_meteor_score = np.mean(all_meteor_scores) 
    print(f"prediction file: {args.pred_path}")
    print(f"METEOR: {mean_meteor_score:.2f}")