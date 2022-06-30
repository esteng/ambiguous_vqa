import argparse
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
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

    for gold, pred in pairs:
        score_dict = rouge.get_scores(pred, gold)[0]

        all_rouge_scores.append(score_dict['rouge-l']['f']) 

    avg_score = np.mean(all_rouge_scores)
    print(f"prediction file: {args.pred_path}")
    print(f"ROUGE-L: {avg_score:.2f}")