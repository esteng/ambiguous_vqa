import csv 
import numpy as np 
import pdb 

np.random.seed(12) 

ann1 = "A2M03MZWZDXKAJ"
ann2 = "A2VIKCIM9TZL22"

with open(f"mturk/full_hit_round_1/{ann1}.csv") as ann1_f, \
     open(f"mturk/full_hit_round_1/{ann2}.csv") as ann2_f: 
    r1_ann1_data = [row for row in csv.DictReader(ann1_f)]
    r1_ann2_data = [row for row in csv.DictReader(ann2_f)]

with open(f"mturk/full_hit_round_2/{ann1}.csv") as ann1_f, \
     open(f"mturk/full_hit_round_2/{ann2}.csv") as ann2_f: 
    r2_ann1_data = [row for row in csv.DictReader(ann1_f)]
    r2_ann2_data = [row for row in csv.DictReader(ann2_f)]


r1_r2_shared_examples = []
r1_ann1_qids = set([row['Input.question_id'] for row in r1_ann1_data])
r2_ann1_qids = set([row['Input.question_id'] for row in r2_ann1_data])
r1_ann2_qids = set([row['Input.question_id'] for row in r1_ann2_data])
r2_ann2_qids = set([row['Input.question_id'] for row in r2_ann2_data])


shared_qids = (r1_ann1_qids & r2_ann2_qids) | (r1_ann2_qids & r2_ann1_qids) 
done = []
for row in r1_ann1_data + r2_ann1_data + r1_ann2_data + r2_ann2_data: 
    if row['Input.question_id'] in shared_qids and row['Input.question_id'] not in done:
        row['HITId'] = row['Input.question_id']
        r1_r2_shared_examples.append(row)
        # done.append(row['Input.question_id'])



with open(f"mturk/full_hit_shared/{ann1}_{ann2}.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=r1_r2_shared_examples[0].keys())
    writer.writeheader()
    writer.writerows(r1_r2_shared_examples)



# rows_by_hit_id = {row['HITId']: [] for row in r1_r2_shared_examples}
# for row in r1_r2_shared_examples:
#     rows_by_hit_id[row['HITId']].append(row)


# for hit_id, rows in rows_by_hit_id.items():
#     if rows[0]['Answer.is_skip'] != rows[1]['Answer.is_skip']:
#         print(f"question: {rows[0]['Input.questionStr']}")
#         print(f"ann {rows[0]['WorkerId']} skipped: {rows[0]['Answer.is_skip']}")
#         print(f"ann {rows[1]['WorkerId']} skipped: {rows[1]['Answer.is_skip']}")

#         print(f"ann {rows[0]['WorkerId']} groups: {rows[0]['Answer.answer_questions']}")
#         print(f"ann {rows[1]['WorkerId']} groups: {rows[1]['Answer.answer_questions']}")
#         pdb.set_trace()

done = []
total = 0
num_skipped = 0
num_ambig = 0
num_redundant = 0
for row in r1_ann1_data + r2_ann1_data + r1_ann2_data + r2_ann2_data:
    if row['Input.question_id'] in done: 
        num_redundant += 1
    else:
        if row['Answer.is_skip'] == 'false':
            num_ambig += 1
        else:
            num_skipped += 1
    total += 1
    done.append(row['Input.question_id'])

print(f"total: {total}, num_skipped: {num_skipped}, num_ambig: {num_ambig}, num_redundant: {num_redundant}")
       