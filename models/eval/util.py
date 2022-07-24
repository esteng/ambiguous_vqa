import argparse
import json 
import re 
import pdb 
from collections import defaultdict

from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-base')

def read_test_data(test_data_path):
    questions = test_data_path.joinpath("questions.json")
    annotations = test_data_path.joinpath("annotations.json")
    questions = json.load(open(questions))
    annotations = json.load(open(annotations))
    return questions['questions'], annotations['annotations']

def read_json_generations(output_path):
    with open(output_path) as f1:
        data = json.load(f1)

    questions = data['questions']
    questions_by_qid = defaultdict(list)
    for q in questions:
        qid = int(q['question_id'].split("_")[0]) 
        questions_by_qid[qid].append(q['question'])
    return questions_by_qid

def read_jsonl_generations(output_path):
    flat_data_by_qid = defaultdict(list)
    data = open(output_path).readlines()
    for line in data:
        batch_data = json.loads(line)
        for qid, generation in zip(batch_data['question_id'], batch_data['speaker_utterances'][0]):
            qid, __ = qid.split("_")
            flat_data_by_qid[int(qid)].append(generation)
    return flat_data_by_qid

def read_generations(output_path):
    str_output_path = str(output_path)
    if str_output_path.endswith(".jsonl"):
        return read_jsonl_generations(output_path)
    elif str_output_path.endswith(".json"):
        return read_json_generations(output_path)
    else:
        raise ValueError(f"Unknown file type: {output_path}")

def match_data(questions, annotations, pred_by_qid, tokenize=True):
    paired = []
    print(len(questions) , len(annotations) ,len(pred_by_qid.keys()))
    for question, annotation in zip(questions, annotations): 
        qid = int(question['question_id'])
        # print(pred_data.keys())
        pred_questions = pred_by_qid[qid]
        gold_question = question['question']
        if tokenize:
            gold_question = tokenizer.tokenize(gold_question)
        for pred_question in pred_questions:
            pred_question = re.sub("<.*?>", "", pred_question) 
            if tokenize:
                pred_question = tokenizer.tokenize(pred_question) 

            paired.append((gold_question, pred_question))
    return paired 
