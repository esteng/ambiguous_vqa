import csv 
import argparse 
import pdb 
import copy 

def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def write_csv(csv_data, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

def unify_rows(row_a, row_b):
    # combine the keys in row a and row b so both have the same keys
    a_not_b_keys = set(row_a.keys()) - set(row_b.keys())
    b_not_a_keys = set(row_b.keys()) - set(row_a.keys())
    for key in a_not_b_keys:
        if key.startswith("Answer"):
            continue
        row_b[key] = "placeholder"
    for key in b_not_a_keys: 
        if key.startswith("Answer"):
            continue
        row_a[key] = "placeholder"
    return row_a, row_b


def merge_csvs(csv_turkle, csv_mturk): 
    ex_a, ex_b = csv_turkle[0], csv_mturk[0]
    if "Input.question_id" in ex_a.keys() and "Input.question_id" in ex_b.keys():
        pivot_key = "Input.question_id"
    else:
        pivot_key = "Input.questionStr"

    to_ret = []
    done = []
    for row_t in csv_turkle:
        for row_m in csv_mturk:
            hit_id_t = row_t['HITId']
            hit_id_m = row_m['HITId']
            if row_t[pivot_key] == row_m[pivot_key]:
                row_t, row_m = unify_rows(row_t, row_m) 
                # if hit_id_t not in done: 
                t_key = f"{row_t['Turkle.Username']}_{hit_id_t}"
                if t_key not in done:
                    to_ret.append(row_t)
                    done.append(t_key)
                    # done.append(hit_id_t)
                m_key = f"{row_m['Turkle.Username']}_{hit_id_m}"
                if m_key not in done:
                    copy_row = copy.deepcopy(row_m)
                    copy_row['HITId'] = hit_id_t
                    to_ret.append(copy_row)
                    done.append(m_key)
                    print(len(done), len(to_ret))

    return to_ret 

if __name__ == "__main__": 
    # define arg parser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--turkle-csv", type=str, required=True)
    parser.add_argument("--mturk-csv", type=str, required=True) 
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    turkle_data = read_csv(args.turkle_csv) 
    mturk_data = read_csv(args.mturk_csv)

    merged_data = merge_csvs(turkle_data, mturk_data)
    write_csv(merged_data, args.out_csv)