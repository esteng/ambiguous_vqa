import argparse
import argparse
from collections import defaultdict
import itertools
import json
import csv 
import pdb
from re import I
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
    fieldnames = ["HITId","HITTypeId","Title","CreationTime","MaxAssignments","AssignmentDurationInSeconds","AssignmentId","WorkerId","AcceptTime","SubmitTime","WorkTimeInSeconds","Input.answerGroups","Input.answerQuestions","Input.imgUrl","Input.questionStr","Answer.answer_groups","Answer.answer_questions","Answer.is_skip","Answer.skipCheck","Answer.skip_reason","Turkle.Username"]
    to_ret = []
    with open(filename) as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            to_ret.append(process_row(row)) 

    return to_ret 

def get_groups(rows, enforce_num_anns, num_anns, annotator_names=None): 
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
    rows_by_hit_id = defaultdict(list)
    for r in rows:
        rows_by_hit_id[r['HITId']].append(r) 
    if enforce_num_anns: 
        rows_by_hit_id = {k: v for k,v in rows_by_hit_id.items() if len(v) == num_anns}
    if annotator_names is not None:
        new_rows = {}
        for k,v in rows_by_hit_id.items(): 
            v = [x for x in v if x['Turkle.Username'] in annotator_names]
            if len(v) == len(annotator_names):
                new_rows[k] = v
        rows_by_hit_id = new_rows
    return rows_by_hit_id

def annotator_report(groups): 
    annotator_lines = defaultdict(list)
    for hit_id, rows in groups.items():
        for row in rows:
            ann = row['Turkle.Username']
            annotator_lines[ann].append(row)

    ann_report = {}
    for ann, rows in annotator_lines.items():
        n_completed = len(rows)
        n_skipped = sum([1 if row['Answer.is_skip'] else 0 for row in rows])
        ann_report[ann] = (n_completed, n_skipped)

    for ann, (completed, skipped) in ann_report.items():
        print(f"Annotator: {ann}, skipped: {skipped}, completed: {completed}")


def skip_agreement(rows_by_hit_id, interact=False, annotator_names=None): # TO DO (TEST)
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

    per_annotator_agreement = defaultdict(lambda: {"correct": 0, "total": 0, "correct_skipped": 0, "correct_unskipped": 0})
    for hit_id, ex_rows in rows_by_hit_id.items(): 
        skips = [ann['Answer.is_skip'] for ann in ex_rows]
        if all(skips) or not any(skips):
            n_agree +=1 
            agree[hit_id] = ex_rows
        else:
            disagree[hit_id] = ex_rows
        for row1 in ex_rows:
            ann1 = row1['Turkle.Username']
            for row2 in ex_rows:
                ann2 = row2['Turkle.Username']
                if annotator_names is not None:
                    if ann1 not in annotator_names or ann2 not in annotator_names:
                        continue 
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

                    else:
                        if interact:
                            pprint([row1, row2], ['Input.imgUrl', 'Input.questionStr', 'Turkle.Username', 'Answer.is_skip'])
                            pdb.set_trace() 
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
        
def group_agreement(rows, enforce_num_anns = False, num_anns=2, interact=False): # TO DO
    rows_by_hit_id = get_groups(rows, enforce_num_anns = enforce_num_anns, num_anns = num_anns) 
    agree, disagree, perc, __ = skip_agreement(rows_by_hit_id) # Agreement, disagreement, percent agreement
    all_groups = [] 
    n_agree, total = 0, 0 # n_agree, total
    group_agree, group_disagree = [], [] # Group agreement, group disagreement

    # Jimen's reworked code

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
                id_sorted_scores[hit_id]['Answer.answer_groups'].append((ann['Turkle.Username'], ann['Answer.answer_groups']))

        else:
            id_sorted_scores[hit_id] = {} 
            id_sorted_scores[hit_id]['Answer.answer_groups'] = []
            for ann in ex_rows: 
                id_sorted_scores[hit_id]['Answer.answer_groups'].append((ann['Turkle.Username'], ann['Answer.answer_groups']))
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
                
                # pdb.set_trace() 
                group_f1 = f1_score(ann1_groups, ann2_groups)
                group_scores[i, ann1_idx, ann2_idx] = group_f1

        
        ann_combos = itertools.combinations(range(len(id_sorted_scores[hit_id]['Answer.answer_groups'])), 2)
        scores_for_avg.append(np.mean([group_scores[i, c[0], c[1]] for c in ann_combos]))

        # id_sorted_scores[hit_id]['GroupAgreement'] = len(group_agree)/(len(group_agree) + len(group_disagree))

        # print(hit_id, id_sorted_scores[hit_id]['GroupAgreement'])
        print(group_scores[i])


        # pprint(rows_by_hit_id[hit_id], ["Input.questionStr", "Turkle.Username", "Answer.answer_groups"])
    print(scores_for_avg)
    return np.mean(scores_for_avg)
########################################

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

def make_lookup(input_row_file): 
    with open(input_row_file) as f1:
        reader = csv.DictReader(f1) 
        lookup = {}
        for row in reader:
            question_str = row['questionStr']
            qid = row['question_id']
            # if question_str in lookup.keys():
                # pdb.set_trace() 
            lookup[question_str] = qid
    return lookup 


def subsample(agree, n_unskipped, n_skipped, annotator_names, lookup): 
    unskipped, skipped = 0, 0
    to_keep = []
    print(len(agree))
    complete = 0
    has_skip_false, has_skip_true = 0, 0
    for hit_id, example_group in agree.items(): 
        # example_rows = [x for x in example_group if x['Turkle.Username'] in annotator_names]
        example_rows = example_group
        example_row = example_rows[0] 
        if example_row['Answer.is_skip']: 
            has_skip_true += 1
        else:
            has_skip_false += 1
        question = example_row['Input.questionStr']
        qid = lookup[question]
        # pdb.set_trace() 
        if example_row['Answer.is_skip']:
            if skipped < n_skipped:
                to_keep.append(qid)
                skipped += 1
        else: 
            if unskipped < n_unskipped:
                to_keep.append(qid)
                unskipped += 1
            else:
                continue

    return to_keep 

def subset_data(input_row_file, output_row_file, to_keep): 
    with open(input_row_file) as f1, \
        open(output_row_file, "w") as f2: 
        reader = csv.DictReader(f1)
        writer = csv.DictWriter(f2, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['question_id'] in to_keep:
                writer.writerow(row)


if __name__ == "__main__": 
    """
    A bit of a hacky file in order to get annotations for the pilot MTurk HIT based on the Turkle HIT.
    Reads in the Turkle results file, figures out which annotations the best annotators agree on, then 
    takes a subset of those annotations, pulls out the corresponding input line, and writes to a csv 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to results csv")
    parser.add_argument("--enforce-num-anns", action='store_true')
    parser.add_argument("--interact", action="store_true")
    parser.add_argument("--n", type=int, default=2, help="number of annotators per example")
    parser.add_argument("--out-path", type=str, default="../csvs/for_mturk.csv")
    parser.add_argument("--input-row-file", type=str, default="../csvs/sorted_by_difference_full.csv")
    parser.add_argument("--n-skipped", type=int, default=28)
    parser.add_argument("--n-unskipped", type=int, default=14)

    parser.add_argument("--annotator-names", type=str, default="esteng,ohussei3")
    args = parser.parse_args()
    args.annotator_names = args.annotator_names.split(",")

    rows = process_csv(args.csv)
    rows_by_hit_id = get_groups(rows, args.enforce_num_anns, args.n, args.annotator_names)
    # annotator_report(rows_by_hit_id)
    (agree, 
    disagree, 
    skip_agree_perc, 
    skip_per_annotator_agreement) = skip_agreement(rows_by_hit_id, 
                                                   interact=args.interact, 
                                                   annotator_names = args.annotator_names) 

    lookup = make_lookup(args.input_row_file)


    to_keep = subsample(agree, n_unskipped = args.n_unskipped, n_skipped= args.n_skipped, annotator_names = args.annotator_names, lookup = lookup)

    subset_data(args.input_row_file, args.out_path, to_keep)
    pdb.set_trace() 

    print(f"annotators agree on skips {skip_agree_perc*100:.2f}% of the time")
    print(f"per_annotator: {skip_per_annotator_agreement}")

    group_agreement = group_agreement(rows, num_anns = args.n, enforce_num_anns=args.enforce_num_anns, interact=args.interact)
