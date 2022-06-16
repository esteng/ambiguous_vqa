from abstract_class import *

def test_BLEU():
    # Bad input -> no result
    assert BLEU_similarity_score()

    # Good input -> results

def test_BERT():
    assert BERT_similarity_score() 

def test_BART():
    assert BART_similarty_score()

def main():
    test_BLEU()
    test_BERT()
    test_BART()

if __name__ == "__main__":
    main()