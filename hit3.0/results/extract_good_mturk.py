import csv 
import argparse

def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def add_turkle_id(csv_data): 
    for i, row in enumerate(csv_data):
        row["Turkle.Username"] = row["WorkerId"]
        csv_data[i] = row
    return csv_data

def write_csv(csv_data, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

def extract_worker_by_id(rows, worker_ids):
    to_ret = []
    for row in rows:
        if row["WorkerId"] in worker_ids:
            to_ret.append(row)
    return to_ret 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--worker-file", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    csv_data = read_csv(args.csv)
    worker_ids = [x.strip() for x in open(args.worker_file).read().split("\n")]
    worker_data = extract_worker_by_id(csv_data, worker_ids)
    worker_data = add_turkle_id(worker_data)
    write_csv(worker_data, args.out_path)