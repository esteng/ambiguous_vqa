from abc import ABC, abstractmethod
import pathlib
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer

import argparse 

class SimilarityClass(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def get_similarity(self):
        pass

class BleuSimilarityScore(SimilarityClass):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1: str, sentence_2: str, num_grams=None) -> str:
        
        if num_grams is None:
            score = sentence_bleu([sentence_1], sentence_2, weights=(1, 0, 0, 0))
        elif num_grams == 1: 
            score = sentence_bleu([sentence_1], sentence_2, weights=(1, 0, 0, 0))
        elif num_grams == 2: 
            score = sentence_bleu([sentence_1], sentence_2, weights=(0, 1, 0, 0))
        elif num_grams == 3: 
            score = sentence_bleu([sentence_1], sentence_2, weights=(0, 0, 1, 0))
        elif num_grams == 4: 
            score = sentence_bleu([sentence_1], sentence_2, weights=(0, 0, 0, 1))
        return score 

class BertSimilarityScore(SimilarityClass):
    def __init__(self, device='cpu'):
        super().__init__()
        self.scorer = BERTScorer(lang='en', device=device)

    def get_similarity(self, sentence_1: str, sentence_2: str) -> str:
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        p, r, f1 = self.scorer.score(format_sent_1, format_sent_2, verbose=False)
        # print(f"BERT Score: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F1.mean().item():.6f}")
        return f1.mean().item()

class BartSimilarityScore(SimilarityClass):
    def __init__(self, type='CNNDM'):
        super().__init__() 
        self.bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
        if type == 'ParaBank':
            self.bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
            path_to_file = pathlib.Path(__file__).resolve().parent.joinpath("BARTScore", "bart.pth")
            self.bart_scorer.load(str(path_to_file))

    def get_similarity(self, sentence_1: str, sentence_2: str, type='CNNDM') -> str:
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        score = self.bart_scorer.score(format_sent_1, format_sent_2, batch_size=1)
        return score 

def main(args):
    if args.model == 'BLEU':

        bleu = BleuSimilarityScore()
        bleu.get_similarity(args.sent_1, args.sent_2)

    if args.model == 'BERT':

        bert = BertSimilarityScore()
        bert.get_similarity("I am good", "You are good")

    if args.model == 'BART':
        
        bart = BartSimilarityScore()
        bart.get_similarity("I am good", "You are good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, dest='model', required=True)
    parser.add_argument("--model-specifics", type=str, dest='specifics', required=False)
    parser.add_argument("--sent-1", type=str, dest='sent_1', required=True)
    parser.add_argument("--sent-2", type=str, dest='sent_2', required=True)
    args = parser.parse_args()

    main(args)

