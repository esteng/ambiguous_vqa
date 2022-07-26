import csv 
from collections import defaultdict
import pdb 
import json 
import ast
import argparse
from xmlrpc.client import Boolean

def main(args):
    dev_size = 30
    all_data = []
    pilot_anns = ["esteng", "A2VIKCIM9TZL22", "A2M03MZWZDXKAJ", "APGX2WZ59OWDN", "A1QUQ0TV9KVD4C", "A2L9763BW12NLA", "ohussei3"]

# first get data from pilot, all of that will go into dev set
    with open("./../hit3.0/results/mturk/pilot_combined_5_anns.csv") as f1:
        reader = csv.DictReader(f1)
        pilot_data = [row for row in reader]

# Get pilot data inputs to get image urls 
    url_lookup = {}
    with open("./../hit3.0/csvs/mturk/pilot_screening.csv") as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            url_lookup[row['questionStr']] = (row['imgUrl'], row['question_id'])


# collect all unskipped 
    pilot_by_qid = defaultdict(list)
    for row in pilot_data: 
        if (row["Turkle.Username"] in pilot_anns and 
            row['Answer.is_skip'] == "false"):
            row['Input.imgUrl'], row['Input.question_id'] = url_lookup[row['Input.questionStr']]
            pilot_by_qid[row['Input.question_id']].append(row)

# go through each question id, get one of the annotations if enough people marked as ambiguous 
    pilot_done = []
    for qid, row_list in pilot_by_qid.items(): 
    # only keep examples for which at least 3 annotators said it was ambiguous 
        if len(row_list) < 3:
            continue
    # go through anns in priority order, with me and the 2 main Mturk workers first 
        for row in row_list:
            for ann in pilot_anns: 
                if row['Input.question_id'] not in pilot_done and row['Turkle.Username'] == ann:
                # check than questions are actually different 
                    if len(set(row['Answer.answer_questions'])) < len(row['Answer.answer_questions']):
                        continue
                    all_data.append(row)
                    pilot_done.append(qid)

# get remaining data from round 1 data 
#with open("mturk/full_hit_round_1/A2M03MZWZDXKAJ.csv") as f1:
    with open("./Mturk_output/csv_results_corrected.csv") as f1:
        reader = csv.DictReader(f1)
        round_1_data = [row for row in reader]

    remaining = dev_size - len(all_data)
    counter = 0 
    examples = []
    f = open("used_examples.json", "w")

    read_obj_6 = open('combined_pilots.json', 'r')
    group_dict = json.load(read_obj_6)

    for row in round_1_data:
        if (row['Answer.is_skip'] != "delete" or row['Answer.is_skip'] != "delete/flag") and counter < remaining:
        
            cluster_group = []
    
            full_cluster_list = ast.literal_eval(row['Answer.answer_groups']) # List of clusters
            for cluster_list in full_cluster_list: # Cluster
                new_cluster = [] # Cluster for cluster
                matched = False
                print(str(cluster_list))
                for answer_dict in cluster_list: # Dict in cluter
                    
                    cur_response = answer_dict["content"]
                    cur_id = answer_dict["id"] 
                    print(str(answer_dict))
           
                    for annotator_response in group_dict[row['Input.question_id']]: 
                        full_new_cluster_list = ast.literal_eval(annotator_response) # New list of clusters
                        for new_cluster_list in full_new_cluster_list: # New clusters
                            for new_answer_dict in new_cluster_list: # New dict
                                if (cur_response == new_answer_dict["content"]):
                                    new_cluster.extend(cluster_list)
                                    new_cluster.remove(answer_dict)
                                    new_cluster.extend(new_cluster_list)
                                    #new_cluster.extend(cluster_list)
                                    matched = True
                                    print('matched')
                                
                if matched == False:
                    new_cluster = cluster_list  
        
                set_cluster = []
                for i in new_cluster:
                    if i not in set_cluster:
                        set_cluster.append(i)
                cluster_group.append(set_cluster)
                
    #print('Combined cluster: \n' + str(cluster_group))
            row['Answer.answer_groups'] = json.dumps(cluster_group)
            row['Answer.answer_questions'] = json.dumps(row['Answer.answer_questions'])

            by_question_id_ambiguity_dict_1 = {}
            by_question_id_ambiguity_dict_2 = {}
            # Ambiguity data
            with open('./../hit3.0/results/categ/csv_clean.csv', 'r') as read_obj_3:
                csv_reader = csv.DictReader(read_obj_3)
                for line in csv_reader:
                    ambiguity_list = line['Answer.skip_reason'].strip('[]"').split(',')
                    by_question_id_ambiguity_dict_1[line['Input.question_id']] = ambiguity_list[0]

    # Cleaned ambiguity data
            with open('./../hit3.0/results/categ/purpose_goal_annotated.csv', 'r') as read_obj_4:
                csv_reader = csv.DictReader(read_obj_4)
                for line in csv_reader:
                    ambiguity_list = line['Answer.skip_reason'].strip('[]"').split(',')
                    by_question_id_ambiguity_dict_2[line['Input.question_id']] = ambiguity_list[0]

            # Setting ambiguity data
            if row['Input.question_id'] in by_question_id_ambiguity_dict_2:
        #jsonl_row['ambiguity_type'] = amb_dict_2.get(line['Input.question_id'])
                row['ambiguity_type'] = by_question_id_ambiguity_dict_2.get(row['Input.question_id'])
            else:
        #jsonl_row['ambiguity_type'] = amb_dict_1.get(line['Input.question_id'])
                row['ambiguity_type'] = by_question_id_ambiguity_dict_1.get(row['Input.question_id'])

            all_data.append(row)
            counter += 1
            examples.append(row['Input.question_id'])
       
    examples_data = {'data': str(examples)}
    json_str = json.dumps(examples_data)
    f.write(json_str)


# write 
    with open("dev_set.csv", "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--repeat-group-data", type=Boolean, dest='repeat', required=False) # Whether or not to repeat group data
    args = parser.parse_args()

    main(args)
