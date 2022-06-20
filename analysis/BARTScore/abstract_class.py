from abc import ABC, abstractmethod
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from bart_score import BARTScorer

import argparse 

class similarity_class(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def get_similarity(self):
        pass

class BLEU_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1: str, sentence_2: str) -> str:
    
        print('BLEU score -> {}'.format(sentence_bleu(sentence_1, sentence_2)))
        print('Individual 1-gram: %f' % sentence_bleu(sentence_1, sentence_2, weights=(1, 0, 0, 0)))
        print('Individual 2-gram: %f' % sentence_bleu(sentence_1, sentence_2, weights=(0, 1, 0, 0)))
        print('Individual 3-gram: %f' % sentence_bleu(sentence_1, sentence_2, weights=(0, 0, 1, 0)))
        print('Individual 4-gram: %f' % sentence_bleu(sentence_1, sentence_2, weights=(0, 0, 0, 1)))


class BERT_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1: str, sentence_2: str) -> str:
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        P, R, F1 = bert_score(format_sent_1, format_sent_2, lang='en', verbose=True)
        print(f"BERT Score: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F1.mean().item():.6f}")

class BART_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1: str, sentence_2: str, type='CNNDM') -> str:
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        if type == 'ParaBank':
            bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='bart.pth')
            bart_scorer.score(format_sent_1, format_sent_2, batch_size=1)
        elif type == 'CNNDM':
            bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
            bart_scorer.score(format_sent_1, format_sent_2, batch_size=1) # generation scores from the first list of texts to the second list of texts.

def main(args):
    if args.model == 'BLEU':

        bleu = BLEU_similarity_score()
        bleu.get_similarity(args.sent_1, args.sent_2)

    if args.model == 'BERT':

        bert = BERT_similarity_score()
        bert.get_similarity("I am good", "You are good")

    if args.model == 'BART':
        
        bart = BART_similarity_score()
        bart.get_similarity("I am good", "You are good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, dest='model', required=True)
    parser.add_argument("--model-specifics", type=str, dest='specifics', required=False)
    parser.add_argument("--sent-1", type=str, dest='sent_1', required=True)
    parser.add_argument("--sent-2", type=str, dest='sent_2', required=True)
    args = parser.parse_args()

    main(args)

