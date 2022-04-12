import argparse
import argparse
from collections import defaultdict
import itertools
import json
import csv 
import pdb
from scipy.optimize import linear_sum_assignment
from tracemalloc import get_tracemalloc_memory 

import numpy as np

def process_row(row):
    columns_to_json = ['Answer.answer_groups', "Answer.answer_questions", "Input.answerGroups", "Input.answerQuestions"]
    for col in columns_to_json:
        # print(row[col])
        row[col] = json.loads(row[col])
    did_skip = True if row['Answer.is_skip'] == "true" else False
    row['Answer.is_skip'] = did_skip
    return row

def process_csv(filename):
    to_ret = []
    with open(filename) as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            to_ret.append(process_row(row)) 

    return to_ret 

def get_groups(rows, enforce_num_anns, num_anns, mturk): 
    """
    Take raw csv rows, and return a dict of lists, with each list containt rows grouped 
    by HIT ID

    Parameters
    ----------
    - rows: List
        A list of csv rows 
    - enforce_num_anns: bool
        if true, only keep HITs with num_anns annotations
    - num_anns: int
        number of annotators participating 
    """
    key="HITId"
    rows_by_hit_id = defaultdict(list)
    for r in rows:
        rows_by_hit_id[r[key]].append(r) 
    if enforce_num_anns: 
        rows_by_hit_id = {k: v for k,v in rows_by_hit_id.items() if len(v) == num_anns}
    return rows_by_hit_id

def annotator_report(groups, mturk): 
    annotator_lines = defaultdict(list)
    if mturk:
        user_key = "WorkerId"
    else:
        user_key = "Turkle.Username"
    for hit_id, rows in groups.items():
        for row in rows:
            ann = row[user_key]
            annotator_lines[ann].append(row)

    ann_report = {}
    for ann, rows in annotator_lines.items():
        n_completed = len(rows)
        n_skipped = sum([1 if row['Answer.is_skip'] else 0 for row in rows])
        ann_report[ann] = (n_completed, n_skipped)

    for ann, (completed, skipped) in ann_report.items():
        print(f"Annotator: {ann}, skipped: {skipped}, completed: {completed}")


def skip_agreement(rows_by_hit_id, interact=False, mturk=False): # TO DO (TEST)
    """
    Compute the percentage of time all annotators agree 
    on whether to skip, and the percentage of times 
    each annotator agrees with each other annotator on
    skipping an example

    Parameters
    ----------
    - rows_by_hit_id: Dict[str, List]
        dict of csv rows with HITId as keys 
    """
    n_agree = 0
    total = 0
    agree = {}
    disagree = {}
    # correct, total 
    if mturk:
        user_key = "WorkerId"
    else:
        user_key = "Turkle.Username"

    per_annotator_agreement = defaultdict(lambda: {"correct": 0, "total": 0, "correct_skipped": 0, "correct_unskipped": 0})
    for hit_id, ex_rows in rows_by_hit_id.items(): 
        skips = [ann['Answer.is_skip'] for ann in ex_rows]
        if all(skips) or not any(skips):
            n_agree +=1 
            agree[hit_id] = ex_rows
        else:
            disagree[hit_id] = ex_rows
        for row1 in ex_rows:
            ann1 = row1[user_key]
            for row2 in ex_rows:
                ann2 = row2[user_key]
                key = f"{ann1}_{ann2}"
                if ann1 == ann2: 
                    continue
                else:
                    if row1['Answer.is_skip'] == row2['Answer.is_skip']: 
                        per_annotator_agreement[key]['correct'] += 1
                        if row1['Answer.is_skip']: 
                            per_annotator_agreement[key]['correct_skipped'] += 1
                        else:
                            per_annotator_agreement[key]['correct_unskipped'] += 1
                        if interact and not row1['Answer.is_skip']:
                            pprint([row1, row2], ['Input.imgUrl', 'Input.questionStr', user_key, 'Answer.is_skip'])
                            pdb.set_trace() 

                    # else:
                    #     if interact:
                    #         pprint([row1, row2], ['Input.imgUrl', 'Input.questionStr', user_key, 'Answer.is_skip'])
                    #         pdb.set_trace() 
                    per_annotator_agreement[key]['total'] += 1


        total += 1
    for k, v in per_annotator_agreement.items():
        per_annotator_agreement[k] = (safe_divide(v['correct'], v['total']), v)
    return agree, disagree, n_agree/total, per_annotator_agreement

def safe_divide(num, denom): 
    try: 
        return num/denom
    except ZeroDivisionError:
        return 0

def f1_helper(group1, group2): 
    """
    Helper function to compute the F1 score
    between two groups 
    """
    ids1 = set([x['id'] for x in group1])
    ids2 = set([x['id'] for x in group2])
    # treat 1 as pred, 2 as true (symmetric so doesn't matter)
    tp = len(ids1 & ids2)
    fp = len(ids1 - ids2) 
    fn = len(ids2 - ids1)

    precision = safe_divide(tp, tp+fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1

def f1_score(groups1, groups2):
    """
    Compute the f1 score between two sets of groups.
    First, compute the F1 score between each of the 
    possible set combinations, then use the 
    Hungarian algorithm to find the maximum assignment,
    i.e. the best alignment between groups in the two sets.
    """
    # what do if groups are different length? 
    # just compute quadriatic, take max  
    p_scores = np.zeros((len(groups1), len(groups2)))
    r_scores = np.zeros((len(groups1), len(groups2)))
    f1_scores = np.zeros((len(groups1), len(groups2)))
    for i, group1 in enumerate(groups1): 
        for j, group2 in enumerate(groups2):  
            p, r, f1 = f1_helper(group1, group2) 
            p_scores[i,j] = p
            r_scores[i,j] = r
            f1_scores[i,j] = f1
    cost_matrix = np.ones_like(f1_scores) * np.max(f1_scores) - f1_scores
    f1_assignment = linear_sum_assignment(cost_matrix) 
    best_f1_scores = f1_scores[f1_assignment]
    return np.mean(best_f1_scores, axis=0)
        
def group_agreement(rows, enforce_num_anns = False, num_anns=2, interact=False, mturk=False): # TO DO
    rows_by_hit_id = get_groups(rows, enforce_num_anns = enforce_num_anns, num_anns = num_anns, mturk=mturk) 
    agree, disagree, perc, __ = skip_agreement(rows_by_hit_id, mturk=mturk) # Agreement, disagreement, percent agreement

    if mturk:
        user_key = "WorkerId"
    else:
        user_key = "Turkle.Username"

    # Group by HitId and then compute pairwise group overlap
    # Have dictionary sorted by example id
    id_sorted_scores = {}

    total_unskipped = 0
    total_skipped = 0
    # Skip skipped examples
    for hit_id, ex_rows in agree.items():
        if ex_rows[0]['Answer.is_skip']:
            total_skipped += 1
            continue

        total_unskipped += 1

        # Put answer_groups into dictionary based on hit id
        if hit_id in id_sorted_scores:
            for ann in ex_rows:
                id_sorted_scores[hit_id]['Answer.answer_groups'].append((ann[user_key], ann['Answer.answer_groups']))

        else:
            id_sorted_scores[hit_id] = {} 
            id_sorted_scores[hit_id]['Answer.answer_groups'] = []
            for ann in ex_rows: 
                id_sorted_scores[hit_id]['Answer.answer_groups'].append((ann[user_key], ann['Answer.answer_groups']))
            # Can input other data such as Input.questionStr, Answer.is_skip, WorkerId, Answer.answer_questions here

    group_agree, group_disagree = [], []

    print(f"total skipped: {total_skipped}")
    print(f"total unskipped: {total_unskipped}")
    # TODO: Jimena: declare this array 
    hit_id = list(id_sorted_scores.keys())[0]
    num_anns = len(id_sorted_scores[hit_id]['Answer.answer_groups']) 
    # group scores: num_annotators, num_annotators, num_annotations
    group_scores = np.zeros((len(id_sorted_scores.keys()), num_anns, num_anns))
    name_to_idx, idx_to_name = {}, {}
    scores_for_avg = []
    for i, hit_id in enumerate(id_sorted_scores.keys()):
        for ann1_idx, (ann1_name, ann1_groups) in enumerate(id_sorted_scores[hit_id]['Answer.answer_groups']):
            for ann2_idx, (ann2_name, ann2_groups) in enumerate(id_sorted_scores[hit_id]['Answer.answer_groups']):
                name_to_idx[ann1_name] = ann1_idx
                name_to_idx[ann2_name] = ann2_idx
                idx_to_name[ann1_idx] = ann1_name
                idx_to_name[ann2_idx] = ann2_name
                if ann1_name == ann2_name: 
                    continue
                
                group_f1 = f1_score(ann1_groups, ann2_groups)
                group_scores[i, ann1_idx, ann2_idx] = group_f1

        
        ann_combos = itertools.combinations(range(len(id_sorted_scores[hit_id]['Answer.answer_groups'])), 2)
        scores_for_avg.append(np.mean([group_scores[i, c[0], c[1]] for c in ann_combos]))

        print(group_scores[i])
    print(scores_for_avg)
    return np.mean(scores_for_avg)

def pprint(rows, fields):
    def stringify(x): 
        if type(x) in [dict, list]: 
            return json.dumps(x, indent=4)
        else:
            return str(x)


    to_print = []
    header = f"{len(rows)} for fields {', '.join(fields)}"
    to_print.append(header) 
    prefix = "\t"
    for row in rows:

        values = [stringify(row[f]) for f in fields]
        to_print.append(f"{prefix}{', '.join(values)}")
    print("\n".join(to_print))
            


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to results csv")
    parser.add_argument("--enforce-num-anns", action='store_true')
    parser.add_argument("--interact", action="store_true")
    parser.add_argument("--n", type=int, default=2, help="number of annotators per example")
    parser.add_argument("--mturk", action="store_true", help="set flag to true if csv is from mturk")
    args = parser.parse_args()

    rows = process_csv(args.csv)
    rows_by_hit_id = get_groups(rows, 
                                args.enforce_num_anns, 
                                args.n,
                                args.mturk)

    annotator_report(rows_by_hit_id, args.mturk)
    agree, disagree, skip_agree_perc, skip_per_annotator_agreement = skip_agreement(rows_by_hit_id, interact=args.interact, mturk=args.mturk) 

    pdb.set_trace() 

    print(f"annotators agree on skips {skip_agree_perc*100:.2f}% of the time")
    print(f"per_annotator: {skip_per_annotator_agreement}")

    group_agreement = group_agreement(rows, num_anns = args.n, enforce_num_anns=args.enforce_num_anns, interact=args.interact)
