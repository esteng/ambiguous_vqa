
import argparse
from pathlib import Path

from nltk.tokenize import word_tokenize
import numpy as np 

from util import read_test_data, read_generations, match_data
from cider.pyciderevalcap.cider.cider import Cider




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    args = parser.parse_args()

    questions, annotations = read_test_data(Path(args.ann_path))
    pred_data = read_generations(Path(args.pred_path))

    pairs = match_data(questions, annotations, pred_data, tokenize=False)

    cider_scorer = Cider()
    all_cider_scores = []

    golds, preds = zip(*pairs)
    fake_golds = {i: [g] for i, g in enumerate(golds)}
    fake_preds = [{"caption": [p], "image_id": i} for i, p in enumerate(preds)]
    cider_score, __ = cider_scorer.compute_score(fake_golds, fake_preds)

    print(f"prediction file: {args.pred_path}")
    print(f"cider: {cider_score:.2f}")