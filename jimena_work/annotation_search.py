# Author: Jimena Guallar-Blasco
# Date Created: 11/28/2024
# Purpose: Convert UDS data into csv format

import json 
from collections import defaultdict
import csv

from decomp import UDSCorpus

uds = UDSCorpus()

# UDS json file
with open('/export/b14/jgualla1/decomp/protoroles.json') as f1:
    protoroles_dict = json.load(f1)

csv_format_list = []

row_counter = 0;

for key in protoroles_dict:
    full_annotation_dict = protoroles_dict[key]
    sentence_ids = full_annotation_dict.keys()
    for sentence_id in sentence_ids:
        if (sentence_id != 'protoroles'):
            sentence = uds[sentence_id].sentence
            node_protorole_dict = full_annotation_dict[sentence_id]
            for node_id in node_protorole_dict:
                # need pred_id
                node1, node2 = node_id.split("%%")
                word1 = uds[sentence_id].head(node1, attrs = ['form', 'lemma'])[1][1]
                pos1 = uds[sentence_id].head(node1, attrs = ['form', 'lemma'])[0]
                word2 = uds[sentence_id].head(node2, attrs = ['form', 'lemma'])[1][1]
                pos2 = uds[sentence_id].head(node2, attrs = ['form', 'lemma'])[0]
                word = f'{word1}%%{word2}'
                pos = f'{pos1}%%{pos2}'

                protorole_dict = node_protorole_dict[node_id]
                attribute_dict = protorole_dict['protoroles']
                for attribute in attribute_dict:
                        
                    cur_dict = {}
                    csv_format_list.append(cur_dict)
                   
                    csv_format_list[row_counter]['sentence_id'] = sentence_id
                    csv_format_list[row_counter]['sentence'] = sentence
                    csv_format_list[row_counter]['node_id'] = node_id
                    csv_format_list[row_counter]['attribute_name'] = attribute
                    csv_format_list[row_counter]['attribute_values'] = attribute_dict[attribute]['value']
                    csv_format_list[row_counter]['annotator_confidence'] = attribute_dict[attribute]['confidence']
                    csv_format_list[row_counter]['word'] = word
                    csv_format_list[row_counter]['position'] = pos
                        
                    row_counter = row_counter + 1
                      

    # filter? i hardly know 'er'!!
csv_format_list_filtered = filter(None, csv_format_list)

with open('protoroles_csv.csv', 'w', newline='') as csvfile:
    fieldnames = ['sentence_id', 'sentence', 'node_id', 'attribute_name', 'attribute_values', 'annotator_confidence', 'word', 'position']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for item in csv_format_list_filtered:
        print(item)
        writer.writerow(item)
