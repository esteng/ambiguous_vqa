# script to prepare training data for running through the model
import json 
import pdb
import re 
import sys 
import pathlib 
import argparse 
import copy

def read_annotation_files(question_file, annotation_file):
    # read in the annotation files
    with open(question_file, 'r') as f:
        questions = json.load(f)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return questions, annotations

def write_annotation_files(questions, annotations, out_path):
    out_path = pathlib.Path(out_path)
    with open(out_path.joinpath("questions.json"), "w") as qf, open(out_path.joinpath("annotations.json"), "w") as af:
        json.dump(questions, qf, indent=4)
        json.dump(annotations, af, indent=4)

def separate(questions, annotations, line_limit=None, exclude_multiple_choice=False):
    question_list = questions['questions']
    annotation_list = annotations['annotations']

    questions_to_write, annotations_to_write = [], []
    for i, (question, annotation) in enumerate(zip(question_list, annotation_list)): 
        if line_limit is not None and i >= line_limit:
            break
        answers = annotation['answers']
        # remove answer_ids
        answer_strs = []
        answers_to_compare = []
        for j, a in enumerate(answers):
            if a['answer'].strip() in answer_strs:
                continue
            answers_to_compare.append(a)
            answer_strs.append(a['answer'].strip())

        for answer in answers_to_compare:
            dummy_answer = [answer]
            if answer['answer'] == annotation['multiple_choice_answer'] and exclude_multiple_choice:
                # don't need to use for now; since it's been trained on 
                continue
            dummy_annotation = copy.deepcopy(annotation)
            dummy_annotation['answers'] = dummy_answer 
            dummy_annotation['multiple_choice_answer'] = dummy_answer[0]['answer']

            questions_to_write.append(question)
            annotations_to_write.append(dummy_annotation)

    new_questions = questions
    new_questions['questions'] = questions_to_write
    new_annotations = annotations
    new_annotations['annotations'] = annotations_to_write

    return new_questions, new_annotations

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='load in the question and annotation files, separate answers into separate lines')
    parser.add_argument("--question-file", type=str, help="path to the question file", required=True)
    parser.add_argument("--annotation-file", type=str, help="path to the annotation file", required=True)
    parser.add_argument("--output-dir", type=str, help="path to the output dir", required=True)
    parser.add_argument("--line-limit", type=int, default=None, help="limit the number of lines to this number")
    parser.add_argument("--exclude-multiple-choice", action="store_true", help="exclude the multiple choice answer")
    args = parser.parse_args()

    questions, annotations = read_annotation_files(args.question_file, args.annotation_file)
    questions, annotations = separate(questions, annotations, args.line_limit, args.exclude_multiple_choice)
    write_annotation_files(questions, annotations, args.output_dir)
