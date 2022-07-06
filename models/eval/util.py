import argparse
import json 
import re 

from nltk.tokenize import word_tokenize

def read_test_data(test_data_path):
    questions = test_data_path.joinpath("questions.json")
    annotations = test_data_path.joinpath("annotations.json")
    questions = json.load(open(questions))
    annotations = json.load(open(annotations))
    return questions['questions'], annotations['annotations']

def read_generations(output_path):
    flat_data_by_qid = {}
    data = open(output_path).readlines()
    for line in data:
        batch_data = json.loads(line)
        for qid, generation in zip(batch_data['question_id'], batch_data['speaker_utterances'][0]):
            flat_data_by_qid[qid] = generation
    return flat_data_by_qid

def match_data(questions, annotations, pred_by_qid, tokenize=True):
    paired = []
    for question, annotation in zip(questions, annotations): 
        qid = question['question_id']
        # print(pred_data.keys())
        pred_question = pred_by_qid[qid]
        gold_question = question['question']
        pred_question = re.sub("<.*?>", "", pred_question)
        if tokenize:
            pred_question = word_tokenize(pred_question)
            gold_question = word_tokenize(gold_question)
        paired.append((gold_question, pred_question))
    # ensure everything is accounted for 
    assert(len(paired) == len(questions) == len(annotations) == len(pred_by_qid.keys()))
    return paired 
