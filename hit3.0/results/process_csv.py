import argparse
import argparse
import json
import csv 
import pdb 

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

def get_groups(rows, num_anns=2): 
    to_ret = []
    for i in range(0, len(rows)-num_anns, num_anns): 
        ex_rows = rows[i:i+num_anns]
        to_ret.append(ex_rows)
    return to_ret 

def skip_agreement(rows, num_anns=2): # TO DO (TEST)
    n_agree = 0
    total = 0
    agree = []
    disagree = []
    for ex_rows in get_groups(rows, num_anns): 
        skips = [ann['Answer.is_skip'] for ann in ex_rows]
        if all(skips) or not any(skips):
            n_agree +=1 
            agree += ex_rows
        else:
            disagree += ex_rows
        total += 1
    return agree, disagree, n_agree/total 
        
def group_agreement(rows, num_anns=2): # TO DO
    agree, disagree, perc = skip_agreement(rows, num_anns) # Agreement, disagreement, percent agreement
    all_groups = [] 
    n_agree, total = 0, 0 # n_agree, total
    group_agree, group_disagree = [], [] # Group agreement, group disagreement

    if False: 
        for ex_rows in get_groups(agree, num_anns): # Rows for each example (annotator x examples)   
            # don't consider skipped examples 
            if ex_rows[0]['Answer.is_skip']: 
                continue 

            do_break = False
            ex_groups = [ann['Answer.answer_groups'] for ann in ex_rows] # for exact agreement (ignore)
            # all_groups.append(ex_groups)

            # Sorting groups 
            for i, ann_groups in enumerate(ex_groups): # Group of annotations from annotators
                for j, group in enumerate(ann_groups): 
                    sorted_group = sorted(group, key=lambda x: x['id'])
                    ex_groups[i][j] = sorted_group

            # Group by HitID

            first_group = ex_groups[0]
            # loop over annotations of an example 
            for ann_groups in ex_groups:
                if do_break:
                    break 

                # loop over groups of annotations 
                for i, gold_group in enumerate(first_group): 
                    # if any are not equal, break TO DO 
                    if do_break:
                        break 
                    # loop over group items 
                    try:
                        other_group = ann_groups[i]
                    except IndexError:
                        group_disagree += ex_rows
                        do_break = True
                        break
                    for j, item in enumerate(gold_group):
                        other_item = other_group[j]
                        if item != other_item: 
                            group_disagree += ex_rows
                            do_break = True
                            break
            n_agree += 1
            group_agree += ex_rows
            total += 1

        pprint(group_disagree, fields = ["Input.questionStr", "Answer.is_skip", "WorkerId", "Answer.answer_questions", "Answer.answer_groups"]) 
        pdb.set_trace() 

########################################
    # Jimen's reworked code

    # Group by HitId and then compute pairwise group overlap
    # Have dictionary sorted by example id
    id_sorted_scores = {}

    # Skip skipped examples
    for ex_rows in get_groups(agree, num_anns):
        if ex_rows[0]['Answer.is_skip']:
            continue

        # do_break = False
        # Put answer_groups into dictionary based on hit id
        if ex_rows[0]['HITId'] in id_sorted_scores:
            for ann in ex_rows:
                id_sorted_scores[ex_rows[0]['HITId']]['Answer.answer_groups'].append(ann['Answer.answer_groups'])

        else:
            id_sorted_scores[ex_rows[0]['HITId']] = {} 
            id_sorted_scores[ex_rows[0]['HITId']]['Answer.answer_groups'] = []
            for ann in ex_rows: 
                id_sorted_scores[ex_rows[0]['HITId']]['Answer.answer_groups'].append(ann['Answer.answer_groups'])
            # Can input other data such as Input.questionStr, Answer.is_skip, WorkerId, Answer.answer_questions here

    # Compute pairwise scores
    # int groups_agree, group_disagree = 0,0

    group_agree, group_disagree = [], []
    for hit_id in id_sorted_scores:
        for i in range(len(id_sorted_scores[hit_id]['Answer.answer_groups'])):
            for j in range(len(id_sorted_scores[hit_id]['Answer.answer_groups'])):
                if i == j: 
                    continue
                # Compute score between two annotators (exact agreement)
                key = (hit_id, i, j)
                # for thing in id_sorted_scores[hit_id]['Answer.answer_groups'][i] 
                    # for list in thing: 
                        # sort list on key
                group_tuple = ([], [])
                idxs = (i, j)
                for t_idx in range(len(group_tuple)):
                    idx = idxs[t_idx]
                    for w, big_list in enumerate(id_sorted_scores[hit_id]['Answer.answer_groups'][idx]): 
                        big_list = sorted(big_list, key=lambda x:x['id']) 
                        group_tuple[t_idx].append(big_list)
                group_a, group_b = group_tuple

      
                if group_a == group_b:
                    group_agree.append(key)
                    # group_agree +=  1 # Will replace with Sorensen's formula
                    # Sorensen's formula: 
                    # DSC = 2|i_set cap j_set|/|i_set|+|j_set|

                else:
                    group_disagree.append(key)

        id_sorted_scores[hit_id]['GroupAgreement'] = len(group_agree)/(len(group_agree) + len(group_disagree))

        print(id_sorted_scores[hit_id]['GroupAgreement'])
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
            


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to results csv")
    parser.add_argument("--n", type=int, default=2, help="number of annotators per example")
    args = parser.parse_args()

    rows = process_csv(args.csv)
    agree, disagree, skip_agree_perc = skip_agreement(rows, num_anns = args.n)

    print(f"annotators agree on skips {skip_agree_perc*100:.2f}% of the time")

    group_agreement = group_agreement(rows, num_anns = args.n)
