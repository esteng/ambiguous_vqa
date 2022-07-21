import argparse
from collections import defaultdict
import itertools
import json
import csv 
from csv import reader
from os import device_encoding
import pdb
from xmlrpc.client import Boolean
from scipy.optimize import linear_sum_assignment
import copy
import ast

from sklearn import cluster

"""
appended csv -> csv
"""

def get_line(line, amb_dict_1, amb_dict_2,username_dict, group_dict, repeat):

    # Setting ambiguity data
    if line['Input.question_id'] in amb_dict_2:
        #jsonl_row['ambiguity_type'] = amb_dict_2.get(line['Input.question_id'])
        line['ambiguity_type'] = amb_dict_2.get(line['Input.question_id'])
    else:
        #jsonl_row['ambiguity_type'] = amb_dict_1.get(line['Input.question_id'])
        line['ambiguity_type'] = amb_dict_1.get(line['Input.question_id'])
    
    # Setting cluster data
    cluster_group = []
    full_cluster_list = ast.literal_eval(line['Answer.answer_groups'])
    for cluster_list in full_cluster_list:
        #print("Cluster: " + str(cluster_list))
        new_cluster = []
        for answer_dict in cluster_list:
           
            
            #print("Answer dict: " + str(answer_dict))
            new_cluster.append(answer_dict)
            response = answer_dict["content"]
            cur_id = answer_dict["id"]
            cur_id_cor = cur_id[:2]
            for annotator_response in group_dict[line['Input.question_id']]:
                full_new_cluster_list = ast.literal_eval(annotator_response)
                for new_cluster_list in full_new_cluster_list:
                #print("List: " + str(new_cluster_list))
                    for new_answer_dict in new_cluster_list:
                        if cur_id_cor == new_answer_dict["id"][:2]:
                            if repeat == True: # Repeats ok if ids different
                                if cur_id != new_answer_dict["id"]: 
                                    new_cluster.append(new_answer_dict)
                            if repeat == False: # No repeats at all
                                if cur_id != new_answer_dict["id"] and response != new_answer_dict["content"]:
                                    new_cluster.append(new_answer_dict)

        cluster_group.append(new_cluster)

    line['Answer.answer_groups'] = str(cluster_group)
    
    return line

def write_csv(to_write, out_path):
    data = []
    fieldnames = ['HITId','HITTypeId','Title','Description','Keywords',
                'Reward','CreationTime','MaxAssignments','RequesterAnnotation',
                'AssignmentDurationInSeconds','AutoApprovalDelayInSeconds',
                'Expiration','NumberOfSimilarHITs','LifetimeInSeconds',
                'AssignmentId','WorkerId','AssignmentStatus','AcceptTime','SubmitTime',
                'AutoApprovalTime','ApprovalTime','RejectionTime',
                'RequesterFeedback','WorkTimeInSeconds','LifetimeApprovalRate',
                'Last30DaysApprovalRate','Last7DaysApprovalRate','Input.imgUrl',
                'Input.questionStr','Input.answerGroups','Input.answerQuestions',
                'Input.question_id','Answer.answer_groups','Answer.answer_questions',
                'Answer.is_skip','Answer.skipCheck','Answer.skip_reason','Approve',
                'Reject','Turkle.Username', 'ambiguity_type']
    with open(out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in to_write:
            
            writer.writerow(row)
            

def sort(data, amb_dict_1, amb_dict_2, username_dict, group_data, repeat, dev):

    delete_count = 0
    flag_count = 0
    delete_flag_count = 0
    dev_delete = 0
    sorted_data = []

    # Sorting out data
    for line in data:
        if line["Answer.skip_reason"] == '"flag"':
            flag_count += 1
        if line["Answer.skip_reason"] == '"delete/flag"':
            delete_flag_count += 1
            continue
        if line["Answer.skip_reason"] == '"delete/flag"' or line["Answer.skip_reason"] == '"delete"':
            delete_count += 1
            continue
        if line["Input.question_id"] in dev:
            dev_delete += 1
            continue
        
        amb_list_1 = []
        amb_list_2 = []
        '''
        if args.include != 'include':
            #print(amb_dict_1)
            #print(line['Input.question_id'])
            amb_list_1 = amb_dict_1[line['Input.question_id']]
            
            if len(amb_list_1) == 1 and amb_list_1[0].strip('"').strip('\\') == 'U':
                delete_count += 1
                continue
            if 'M/A' in amb_list_1 or 'M/B' in amb_list_1:
                delete_count += 1
                continue
            
            amb_list_2 = amb_dict_2.get(line['Input.question_id'])
            if amb_list_2 != None:
                if len(amb_list_2) == 1 and amb_list_2[0].strip('"').strip('\\') == 'U':
                    delete_count += 1
                    continue
                if amb_list_2[0].strip('"').strip('\\') == 'skip':
                    delete_count += 1
                    continue 
                if 'M/A' in amb_list_2 or 'M/B' in amb_list_2:
                    delete_count += 1
                    continue
        '''
        
        sorted_data.append(line)

    # Priting data statistics
    print("Data stats: ")
    print("\tExamples deleted: " + str(delete_count))
    print("\tExamples flagged (kept and deleted): " + str(flag_count))
    print("\tExamples deleted because in dev: " + str(device_encoding))
    exit 

    # Writing data
    to_write = [get_line(l, amb_dict_1, amb_dict_2, username_dict, group_data, repeat) for l in sorted_data]
    write_csv(to_write, args.out_path)

# Main function
def main(args):

    by_question_id_ambiguity_dict_1 = {}
    by_question_id_ambiguity_dict_2 = {}
    by_hitid_annotation_dict = {}
    by_question_id_group_dict = {}
    # Ambiguity data
    with open(args.ambiguity_1) as read_obj_3:
        csv_reader = csv.DictReader(read_obj_3)
        for row in csv_reader:
            ambiguity_list = row['Answer.skip_reason'].strip('[]"').split(',')
            by_question_id_ambiguity_dict_1[row['Input.question_id']] = ambiguity_list[0]

    # Cleaned ambiguity data
    with open(args.ambiguity_2) as read_obj_4:
        csv_reader = csv.DictReader(read_obj_4)
        for row in csv_reader:
            ambiguity_list = row['Answer.skip_reason'].strip('[]"').split(',')
            by_question_id_ambiguity_dict_2[row['Input.question_id']] = ambiguity_list[0]
            
    
    # Original annotator data
    with open(args.original) as read_obj_5:
        csv_reader = csv.DictReader(read_obj_5)
        for row in csv_reader:
            if row['question_id'] not in by_hitid_annotation_dict:
                by_hitid_annotation_dict[row['question_id']] = []
                temp_dict = {}
                temp_dict['Username'] = 'To do'
                temp_dict['Answer_groups'] = row['answerGroups']
                by_hitid_annotation_dict[row['question_id']].append(temp_dict)
            else:
                temp_dict = {}
                temp_dict['Username'] = 1111
                temp_dict['Answer_groups'] = row['answerGroups']
                by_hitid_annotation_dict[row['question_id']].append(temp_dict)
        #print(by_hitid_annotation_dict)


    with open(args.original_groups) as read_obj_6:
        by_question_id_group_dict = json.load(read_obj_6)

    
    with open(args.dev) as read_obj_7:
        examples_in_dev_set_dict = json.load(read_obj_7)
        examples_in_dev_set = examples_in_dev_set_dict['data']
        

    data = []
    
    # Cleaned data
    with open(args.cleaned_1) as read_obj_1:
        csv_reader = csv.DictReader(read_obj_1)
        for row in csv_reader:
            data.append(row)
    sort(data, by_question_id_ambiguity_dict_1, by_question_id_ambiguity_dict_2, by_hitid_annotation_dict, by_question_id_group_dict, args.repeat, examples_in_dev_set)
    
    # More cleaned data
    with open(args.cleaned_2) as read_obj_2:
        csv_reader = csv.DictReader(read_obj_2)
        for row in csv_reader:
            data.append(row)
    sort(data, by_question_id_ambiguity_dict_1, by_question_id_ambiguity_dict_2, by_hitid_annotation_dict, by_question_id_group_dict, args.repeat, examples_in_dev_set)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-data-1", type=str, dest='cleaned_1', required=True, help='csv data') # Clean data, first pass 
    parser.add_argument("--cleaned-data-2", type=str, dest='cleaned_2', required=False, help='csv data to be appended') # Clean data, second pass
    parser.add_argument("--ambiguity-data-1", type=str, dest='ambiguity_1', required=False, help='ambiguity data to be consolidated') # Ambiguity data first pass
    parser.add_argument("--ambiguity-data-2", type=str, dest='ambiguity_2', required=False) # Ambiguity data second pass
    parser.add_argument("--original-data", type=str, dest='original', required=False) # Original annotator data (with annotator usernames)
    parser.add_argument("--original-group-data", type=str, dest='original_groups', required=False) # Original groups data
    parser.add_argument("--repeat-group-data", type=Boolean, dest='repeat', required=False) # Whether or not to repeat group data
    parser.add_argument("--dev-data", type=str, dest='dev', required=False) # Data in dev set
    parser.add_argument("--out-path", type=str, dest='out_path', required=True, help='out path') # Output file

    parser.add_argument("--include", type=str, dest='include', required=False)
    args = parser.parse_args()

    main(args)