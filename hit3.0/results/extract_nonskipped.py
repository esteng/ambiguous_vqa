import csv 
import sys 
import json 
from collections import defaultdict

trusted_anns = ["esteng", "A2VIKCIM9TZL22", "A2M03MZWZDXKAJ", "APGX2WZ59OWDN", "A1QUQ0TV9KVD4C", "A2L9763BW12NLA", "ohussei3"]
anon_alias = {"esteng": "A001", "A2VIKCIM9TZL22": "A002", "A2M03MZWZDXKAJ": "A003", 
              "APGX2WZ59OWDN": "A004", "A1QUQ0TV9KVD4C": "A005", "A2L9763BW12NLA": "A006", 
              "ohussei3": "A007"} 


def preprocess_pilot(pilot_anns):
    # Get pilot data inputs to get image urls 
    url_lookup = {}
    with open("../csvs/mturk/pilot_screening.csv") as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            url_lookup[row['questionStr']] = (row['imgUrl'], int(row['question_id']))
    # collect all unskipped 
    pilot_to_ret = []
    for i, row in enumerate(pilot_anns): 
        if (row["Turkle.Username"] in trusted_anns and 
            row['Answer.is_skip'] == "false"):
            row['WorkerId'] = row['Turkle.Username']
            row['Input.imgUrl'], row['Input.question_id'] = url_lookup[row['Input.questionStr']]
            assert(row['Input.question_id'] != "placeholder")
            pilot_to_ret.append(row)
    return pilot_to_ret

def extract(csv_path, out_path):
    with open(csv_path) as f1:
        reader = csv.DictReader(f1)
        data = [row for row in reader]
    
    data = [x for x in data if x['Answer.is_skip'] == "false"]
    data = [x for x in data if (x['WorkerId'] in trusted_anns or x["Turkle.Username"] in trusted_anns)]
    if "pilot" in csv_path: 
        data = preprocess_pilot(data)

    # anonymize data 
    for i, row in enumerate(data): 
        try:
            row['WorkerId'] = anon_alias[json.loads(row['WorkerId'])]
            row['Turkle.Username'] = row['WorkerId']
        except KeyError:
            row['WorkerId'] = anon_alias[row['Turkle.Username']]
            row['Turkle.Username'] = row['WorkerId']
        except json.JSONDecodeError:
            row['WorkerId'] = anon_alias[row['WorkerId']]
            row['Turkle.Username'] = row['WorkerId']

        data[i] = row 

    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])