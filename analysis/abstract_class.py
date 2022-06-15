from abc import ABC, abstractmethod
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from bart.pth import BARTScorer as bart_score

class similarity_class(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def get_similarity(self):
        pass

class BLEU_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1, sentence_2, gram=None):
        format_sent_1 = sentence_1.split()
        format_sent_2 = sentence_2.split()
        if gram == None:
            print('BLEU score -> {}'.format(sentence_bleu(format_sent_1, format_sent_2)))
        elif gram == 1:
            print('Individual 1-gram: %f' % sentence_bleu(format_sent_1, format_sent_2, weights=(1, 0, 0, 0)))
        elif gram == 2:
            print('Individual 2-gram: %f' % sentence_bleu(format_sent_1, format_sent_2, weights=(0, 1, 0, 0)))
        elif gram == 3:
            print('Individual 3-gram: %f' % sentence_bleu(format_sent_1, format_sent_2, weights=(0, 0, 1, 0)))
        elif gram == 4:
            print('Individual 4-gram: %f' % sentence_bleu(format_sent_1, format_sent_2, weights=(0, 0, 0, 1)))

class BERT_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1, sentence_2):
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        P, R, F1 = bert_score(format_sent_1, format_sent_2, lang='en', verbose=True)
        print(f"BERT Score: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F1.mean().item():.6f}")


class BART_similarity_score(similarity_class):
    def __init__(self):
        super().__init__()

    def get_similarity(self, sentence_1, sentence_2, type='ParaBank'):
        format_sent_1 = [sentence_1]
        format_sent_2 = [sentence_2]
        if type == 'ParaBank':
            bart_scorer = bart_score(device='cuda:0', checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='bart.pth')
            bart_scorer.score(format_sent_1, format_sent_2, batch_size=1)
        elif type == 'CNNDM':
            bart_scorer = bart_score(device='cuda:0', checkpoint='facebook/bart-large-cnn')
            bart_scorer.score(format_sent_1, format_sent_2, batch_size=1) # generation scores from the first list of texts to the second list of texts.

<<<<<<< HEAD

=======
>>>>>>> 7a45be945493df0ebde3aa4bfecc871ef6d5c1c4
bleu = BLEU_similarity_score()
bleu.get_similarity("I am good", "You are good")

bert = BERT_similarity_score()
bert.get_similarity("I am good", "You are good")

bart = BART_similarity_score()
bart.get_similarity("I am good", "You are good")



