import argparse
from pathlib import Path
import pdb 


from bert_score import BERTScorer
import numpy as np 

from util import read_test_data, read_generations, match_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    args = parser.parse_args()

    questions, annotations = read_test_data(Path(args.ann_path))
    pred_data = read_generations(Path(args.pred_path))

    pairs = match_data(questions, annotations, pred_data, tokenize=False)

    scorer = BERTScorer(lang='en', device='cuda:0')
    refs, preds = zip(*pairs)
    bs = scorer.score(refs, preds) 
    scores = []
    for batch in bs:
        for score in batch:
            scores.append(score)
    avg_score = np.mean(scores)
    print(f"prediction file: {args.pred_path}")
    print(f"BERT Score: {avg_score:.2f}")