import csv 
import numpy as np 

np.random.seed(12) 

ann1 = "A2M03MZWZDXKAJ"
ann2 = "A2VIKCIM9TZL22"

with open(f"mturk/full_hit_round_1/{ann1}.csv") as ann1_f, \
     open(f"mturk/full_hit_round_1/{ann2}.csv") as ann2_f: 
     ann1_data = [row for row in csv.DictReader(ann1_f)]
     ann2_data = [row for row in csv.DictReader(ann2_f)]

ann1_skipped, ann1_nonskipped, ann2_skipped, ann2_nonskipped = [], [], [], []
for row in ann1_data:
    if row['Answer.is_skip'] in ["False", "false", False]: 
        ann1_nonskipped.append(row)
    else:
        ann1_skipped.append(row)

for row in ann2_data:
    if row['Answer.is_skip'] in ["False", "false", False]: 
        ann2_nonskipped.append(row)
    else:
        ann2_skipped.append(row)

print("ann1 didn't skip:", len(ann1_nonskipped))
print("ann2 didn't skip:", len(ann2_nonskipped))

# sample the same number of unskipped rows 
ann1_skip_sample = np.random.choice(ann1_skipped, len(ann1_nonskipped), replace=False).tolist()
ann2_skip_sample = np.random.choice(ann2_skipped, len(ann2_nonskipped), replace=False).tolist()


print("ann1 skipped sample:", len(ann1_skip_sample))
print("ann2 skipped sample:", len(ann2_skip_sample))

ann1_to_add = ann1_nonskipped + ann1_skip_sample
ann2_to_add = ann2_nonskipped + ann2_skip_sample

ann1_ids_to_add = [row['Input.question_id'] for row in ann1_to_add]
ann2_ids_to_add = [row['Input.question_id'] for row in ann2_to_add]

# read the original data for ann1, read the original data for ann2
with open(f"../csvs/mturk_full/round_1/{ann1}.csv") as ann1_og_f, \
    open(f"../csvs/mturk_full/round_1/{ann2}.csv") as ann2_og_f: 
    ann1_og_data = [row for row in csv.DictReader(ann1_og_f)]
    ann2_og_data = [row for row in csv.DictReader(ann2_og_f)]


ann1_og_subset = [row for row in ann1_og_data if row['question_id'] in ann1_ids_to_add]
ann2_og_subset = [row for row in ann2_og_data if row['question_id'] in ann2_ids_to_add]

# read the new data, mix in the original data 
with open(f"../csvs/mturk_full/round_2/{ann1}.csv") as ann1_new_f, \
     open(f"../csvs/mturk_full/round_2/{ann2}.csv") as ann2_new_f: 
    ann1_new_data = [row for row in csv.DictReader(ann1_new_f)]
    ann2_new_data = [row for row in csv.DictReader(ann2_new_f)]

# give ann2's annotations to ann1, ann1's annotations to ann2
ann1_new_data = ann1_new_data + ann2_og_subset
ann2_new_data = ann2_new_data + ann1_og_subset

print(f"{ann1} new data: {len(ann1_new_data)}, with {len(ann2_og_subset)} from ann2")
print(f"{ann2} new data: {len(ann2_new_data)}, with {len(ann1_og_subset)} from ann1")

# mix it up 
np.random.shuffle(ann1_new_data)
np.random.shuffle(ann2_new_data)

# write the new files 
with open(f"../csvs/mturk_full/round_2_plus_round_1_results/{ann1}.csv", "w") as ann1_new_f, \
     open(f"../csvs/mturk_full/round_2_plus_round_1_results/{ann2}.csv", "w") as ann2_new_f:
     # write ann1_new_data to ann1_new_f
        writer1 = csv.DictWriter(ann1_new_f, fieldnames=ann1_new_data[0].keys())
        writer1.writeheader()
        writer1.writerows(ann1_new_data)

        writer2 = csv.DictWriter(ann2_new_f, fieldnames=ann2_new_data[0].keys())
        writer2.writeheader()
        writer2.writerows(ann2_new_data)