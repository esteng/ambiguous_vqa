from abc import ABCMeta, abstractmethod
from nltk.translate.bleu_score import sentence_blue
from bert_score import score

class similarity_class(ABC):
    def __init__(self, original_sentence, test_sentence):
        self.sent_1 = original_sentence
        self.sent_2 = test_sentence

    @abstractmethod
    def get_similarity(self):
        pass

class BLEU(similarity_class):
    def __init__(self, original_sentence, test_sentence):
        super().__init__(original_sentence, test_sentence)

    def get_similarity(self):
        #reference = 
        #candidate = 
        print('BLEU score -> {}'.format(sentence_bleu(refernce, candidate)))

class BERT(similarity_class):
    def __init__(self, original_sentence, test_sentence):
        super().__init__(original_sentence, test_sentence)

    def get_similarity(self):
        return similarity

class BART(similarity_class):
    def __init__(self, original_sentence, test_sentence):
        super().__init__(original_sentence, test_sentence)

    def get_similarity(self):
        return similarity
