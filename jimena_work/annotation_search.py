import json 
from collections import defaultdict
import csv

from decomp import UDSCorpus

uds = UDSCorpus()

with open('/export/b14/jgualla1/decomp/protoroles.json') as f1:
    protoroles_dict = json.load(f1)

#annotations_dict = defaultdict(list)
    csv_format_list = []

    row_counter = 0;

    for key in protoroles_dict:
        full_annotation_dict = protoroles_dict[key]
        sentence_ids = full_annotation_dict.keys()
        for sentence_id in sentence_ids:
            # sentence_id
            #print(sentence_id)
            if (sentence_id != 'protoroles'):
                sentence = uds[sentence_id].sentence
                # sentence
                #print(sentence) # Sentence
                node_protorole_dict = full_annotation_dict[sentence_id]
                for node_id in node_protorole_dict:
                    protorole_dict = node_protorole_dict[node_id]
                    attribute_dict = protorole_dict['protoroles']
                    for attribtue in attribute_dict:
                        
                        cur_dict = {}
                        csv_format_list.apppend(cur_dict)
                        # is there a better way to do this? you need an append-ectomy
                        csv_format_list[row_counter]['sentence_id'] = sentence_id
                        csv_format_list[row_counter]['sentence'] = sentence
                        csv_format_list[row_counter]['node_id'] = protorole
                        csv_format_list[row_counter]['attribute_name'] = attribute
                        csv_format_list[row_counter]['attribute_values'] = attribute[1]
                        csv_format_list[row_counter]['annotator_confidence'] = attribute[0]
                        csv_format_list[row_counter]['word'] = 'WORD'
                        
                        row_counter = row_counter + 1
                        

    # filter? i hardly know 'er'!!
    csv_format_list_filtered = filter(None, csv_format_list)

with open('protoroles_csv.csv', 'w', newline='') as csvfile:
    fieldnames = ['sentence_id', 'sentence', 'node_id', 'attribute_name', 'attribute_values', 'annotator_confidence', 'word']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for item in csv_format_list_filtered:
        writer.writerow(item)
