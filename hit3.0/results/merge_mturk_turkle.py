import csv 
import argparse 
import pdb 
import copy
import json 

from process_csv import process_pilot_row

def read_csv(csv_path, pilot=False):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        if pilot:
            to_ret = []
            for row in reader:
                to_ret += process_pilot_row(row, as_json=True)
            return to_ret 
        else:
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


def merge_csvs(csv_turkle, csv_mturk, done=None): 
    ex_a, ex_b = csv_turkle[0], csv_mturk[0]
    if "Input.question_id" in ex_a.keys() and "Input.question_id" in ex_b.keys():
        pivot_key = "Input.question_id"
    else:
        pivot_key = "Input.questionStr"

    to_ret = []
    if done is None:
        done = []
    done_t = []
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
                    done_t.append(t_key)
                    if t_key == "esteng_476670":
                        print("esteng hit")
                    # done.append(hit_id_t)
                m_key = f"{row_m['Turkle.Username']}_{hit_id_m}"
                if m_key not in done:
                    copy_row = copy.deepcopy(row_m)
                    copy_row['HITId'] = hit_id_t
                    to_ret.append(copy_row)
                    done.append(m_key)
                    # print(len(done), len(to_ret))
    print(done_t)
    pdb.set_trace()
    return to_ret, done

if __name__ == "__main__": 
    # define arg parser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--turkle-csv", type=str, required=True)
    parser.add_argument("--mturk-csvs", type=str, required=True) 
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    turkle_data = read_csv(args.turkle_csv) 
    mturk_csvs = args.mturk_csvs.split(",")
    merged_data = turkle_data
    big_done = []
    mturk_data = []
    for mturk_csv in mturk_csvs:
        mturk_data += read_csv(mturk_csv, pilot=args.pilot)

    print(len(mturk_data))
    merged_data, done = merge_csvs(turkle_data, mturk_data, big_done)
    big_done += done
    # merged_data += to_merge
    write_csv(merged_data, args.out_csv)