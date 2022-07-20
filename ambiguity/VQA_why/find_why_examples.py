# finding why examples

import json


f = open('../../../../../export/b02/jgualla1/VQA/v2_OpenEnded_mscoco_train2014_questions.json') 
j = open('../../../../../export/b02/jgualla1/VQA/v2_mscoco_train2014_annotations.json')

q_data = json.load(f)
a_data = json.load(j)

question_data = q_data['questions']
annotation_data = a_data['annotations']

# Finding 'why' questions
why_question_data_sorted_by_id = {}

for question_dict in question_data:
    # Finding reasoning only examples
    if 'Why' in question_dict['question']:
        why_question_data_sorted_by_id[question_dict['question_id']] = question_dict

print(len(why_question_data_sorted_by_id))

f1 = open('examples.txt', 'w')

example_count = 100 # Number of examples to be included
count = 0

final_why_question_data = {}

# Finding annotation information for 'why' questions
for annotation_dict in annotation_data:
    if annotation_dict['question_id'] in why_question_data_sorted_by_id:
        
        count += 1
        
        # Add question data to dict
        temp_dict  = {'question_id': 0, 
                'question': '',
                'non_repeat_answers': []}
        temp_dict['question_id'] = annotation_dict['question_id']
        temp_dict['question'] = why_question_data_sorted_by_id[annotation_dict['question_id']]['question']

        
        # Remove repeated answers
        repeat_answer = []
        for answer in annotation_dict['answers']:
            repeat_answer.append(answer['answer'])
        non_repeat_answers = set(repeat_answer)

        # Add answer to temp dict
        answer_count = 0
        for answer in non_repeat_answers:
            temp_dict['non_repeat_answers'].append([{'id': answer_count,'content': answer}])
            answer_count += 1
        final_why_question_data[annotation_dict['question_id']] = temp_dict
    if example_count == count:
        break

# Write dict to json file
with open('examples.json', 'w') as k:
    json.dump(final_why_question_data, k)

