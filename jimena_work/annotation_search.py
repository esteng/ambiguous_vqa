import json 
from collections import defaultdict

from decomp import UDSCorpus

uds = UDSCorpus()

with open('/export/b14/jgualla1/decomp/protoroles.json') as f1:
    protoroles_dict = json.load(f1)

#annotations_dict = defaultdict(list)

with open('protoroles_csv', 'w', newline='') as csvfile:
    fieldnames = ['Sentence Id', 'Sentence', 'Node Id', 'Node', 'Annotator Scores', 'Annotator Confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writerheader()

    csv_format_list = []

    for key in protoroles_dict:
        full_annotation_dict = protoroles_dict[key]
        sentence_ids = full_annotation_dict.keys()
        for sentence_id in sentence_ids:
            print(sentence_id) # Sentence ID
            if (sentence_id != 'protoroles'):
                sentence = uds[sentence_id].sentence

                print(sentence) # Sentence
                protorole_dict = full_annotation_dict[sentence_id]
                for protorole in protorole_dict:
                    csv_format_list.append(sentence_id)
                    csv_format_list.append(sentence)
                    csv_format_list.append(protorole)
                    csv_format_list.append(protorole_dict[protorole])
                    csv_format_list.append(protorole_dict[protorole])
                    print(protorole) # Protorole ID
                    print(protorole_dict[protorole]) # Protorole data (annotator score/confidence)

