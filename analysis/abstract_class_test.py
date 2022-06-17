from abstract_class import *

def test_BLEU():
    # Bad input -> no result
    # The dog sleeps well vs. The dog jumps well
    assert BLEU_similarity_score('I am good', 'You are good') == 1.4637115948630222e-231 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 1) == 0.416667
    assert BLEU_similarity_score('I am good', 'You are good', 2) == 0.000000 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 3) == 0.000000
    assert BLEU_similarity_score('I am good', 'You are good', 4) == 0.000000

    # I am good vs. You are good
    assert BLEU_similarity_score('I am good', 'You are good') == 1.4637115948630222e-231 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 1) == 0.416667
    assert BLEU_similarity_score('I am good', 'You are good', 2) == 0.000000 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 3) == 0.000000
    assert BLEU_similarity_score('I am good', 'You are good', 4) == 0.000000

    # Their mosquito nets did not protect vs. The behives did a lot of damage
    assert BLEU_similarity_score('I am good', 'You are good') == 1.3729916628506288e-231 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 1) == 0.322581
    assert BLEU_similarity_score('I am good', 'You are good', 2) == 0.000000 # 0N
    assert BLEU_similarity_score('I am good', 'You are good', 3) == 0.000000
    assert BLEU_similarity_score('I am good', 'You are good', 4) == 0.000000

   
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