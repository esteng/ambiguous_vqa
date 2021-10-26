import argparse 
from tqdm import tqdm 
import numpy as np 
import sys 
import pdb 
import re 
import json
from pathlib import Path
import time 

import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, get_context
import psutil 


from sklearn.cluster import KMeans

def get_glove(path):
    lookup = {}
    with open(path) as f1:
        for line in f1:
            splitline = re.split("\s+", line.strip())
            word = splitline[0]
            try:
                vec = np.array(splitline[1:], dtype=float) 
                # vec = [float(x) for x in splitline[1:]]
            except ValueError:
                continue
            lookup[word] = vec
    return lookup 

def get_kmeans_score(annotation_set, glove_lookup, kmeans_wrapper_dict, penalty_factor = 2, reduction='mean'):
    # TODO: remove later! 
    if glove_lookup is None:
        keys = [w  for ann in annotation_set for w in ann]
        glove_lookup = {k: np.random.uniform(size=300) for k in keys}

    annotation_set_as_vec = []
    for i, ann in enumerate(annotation_set):
        ann = ann.split(" ")
        ann = [w for w in ann if w in glove_lookup.keys()]
        if len(ann) == 0:
            continue

        vecs = np.array([glove_lookup[word] for word in ann])
        if reduction == 'mean': 
            # mean pool
            annotation_set_as_vec.append(np.mean(vecs, axis=0) )
        elif reduction == 'max':
            # max pool
            annotation_set_as_vec.append(np.max(vecs, axis=0) )
        else:
            print(f'invalid reduction {reduction}')
            exit()

    if len(annotation_set_as_vec) == 0: 
        return None

    annotation_set_as_vec = np.array(annotation_set_as_vec).reshape(-1, 300)
    scores_data = []

    for k in range(1, annotation_set_as_vec.shape[0]):
        kmeans_wrapper = kmeans_wrapper_dict[k]    
        # run kmeans
        kmeans = kmeans_wrapper.fit(annotation_set_as_vec) 
        centers = kmeans.predict(annotation_set_as_vec) 
        num_centers = len(set(centers))
        # intertia is sum of squared distances of samples to their closest cluster center 
        inertia = kmeans.inertia_
        # penalty depends on how many centers you used compared to how many examples you have 
        penalty = penalty_factor * (num_centers-1)/ (len(annotation_set)-1) 
        # we want balanced clusters
        avg = int(len(annotation_set) / num_centers)
        num_per_center = {c_name: sum([1 for c in centers if c == c_name]) for c_name in centers}
        diffs = [abs(avg - num_per_center[c]) for c in centers]
        difference_penalty = sum(diffs)
        score = inertia + penalty + difference_penalty 

        # score = kmeans.inertia_ + num_centers ** penalty_factor
        # high intertia means dispersed, low means fits well to the number of clusters
        scores_data.append((score, inertia, penalty, difference_penalty, num_centers))
        # if you hit zero inertia, no sense in going further 
        if kmeans.inertia_ < 1e-16:
            break

    score_array = np.array(scores_data)
    try:
        min_row = np.argmin(score_array[:,0])
        return score_array[min_row, :]
    except IndexError:
        return -1, -1, -1, -1, -1

class ArgumentWrapper:
    def __init__(self, annotation, question_lookup, glove_lookup, kmeans_wrapper_dict, penalty_factor, reduction):
        self.annotation = annotation
        annotation_set = [x['answer'] for x in annotation['answers'] if x['answer_confidence'] == 'yes']
        words = [w for ann in annotation_set for w in ann.split(" ")]
        if glove_lookup is None:
            self.glove_lookup = dict()
        else:
            self.glove_lookup = {w: glove_lookup[w]  for w in words if w in glove_lookup.keys()} 
        self.question = question_lookup[annotation['question_id']]
        self.kmeans_wrapper_dict = kmeans_wrapper_dict
        self.penalty_factor = penalty_factor
        self.reduction = reduction 

def make_training_example(arg_wrapper):
    #print(f"cpu num {psutil.Process().cpu_num()}") 
    annotation = arg_wrapper.annotation
    glove_lookup = arg_wrapper.glove_lookup 
    kmeans_wrapper_dict = arg_wrapper.kmeans_wrapper_dict 
    penalty_factor = arg_wrapper.penalty_factor 
    question = arg_wrapper.question 
    reduction = arg_wrapper.reduction

    image_id = annotation['image_id']
    annotation_set = [x['answer'] for x in annotation['answers'] if x['answer_confidence'] == 'yes']
    kmeans_data =  get_kmeans_score(annotation_set, glove_lookup, kmeans_wrapper_dict, penalty_factor, reduction)    

    new_annotation = {"question": question,
                      "question_id": annotation['question_id'],
                      "image_id": image_id,
                      "annotation_set": annotation_set,
                      "kmeans_score": -1,
                      "inertia": -1,
                      "num_clusters": -1,
                      "penalty": -1,
                      "difference_penalty": -1}

    if kmeans_data is None:
        return new_annotation

    kmeans_score, inertia, penalty, difference_penalty, num_clusters = kmeans_data
    new_annotation['kmeans_score'] = kmeans_score
    new_annotation['inertia'] = inertia
    new_annotation['penalty'] = penalty
    new_annotation['difference_penalty'] = difference_penalty
    new_annotation['num_clusters'] = num_clusters 

    return new_annotation

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--glove-path", type=str, default = "/srv/local1/estengel/resources/glove.840B.300d.txt", help="path to glove")
    parser.add_argument("--ann-path", type=str, default = "/srv/local2/estengel/annotator_uncertainty/vqa/v2_mscoco_val2014_annotations.json", help="path to glove")
    parser.add_argument("--question-path", type=str, default = "/srv/local2/estengel/annotator_uncertainty/vqa/v2_OpenEnded_mscoco_val2014_questions.json", help="path to glove")
    parser.add_argument("--penalty-factor", type=float, default=3.0, help="penalty factor per additional cluster")
    parser.add_argument("--num-workers", default=1, type=int, help="if multiprocessing, number of workers")
    parser.add_argument("--num-examples", default=-1, type=int, help="number of examples to process")
    parser.add_argument("--reduction", type=str, help="reduction for pooling glove", choices=["mean", "max"], default="mean")
    # parser.add_argument("--penalty-type")
    parser.add_argument("--out-path", type=str, default = "/srv/local2/estengel/annotator_uncertainty/vqa/")
    args = parser.parse_args() 

    start_time = time.time()
    kmeans_wrapper_dict = {k: KMeans(n_clusters=k, random_state=12) for k in range(0, 10)}
    print(f"Building glove lookup...")
    #glove_lookup = None
    glove_lookup = get_glove(args.glove_path) 
    glove_time = time.time() 
    print(f"reading glove took {glove_time - start_time:.2f}")

    print(f"Reading data...")
    with open(args.ann_path) as f1, open(args.question_path) as f2:
        all_anns = json.load(f1)
        all_quests = json.load(f2)
    question_lookup = {e['question_id']: e for e in all_quests['questions']}

    print(f"building arg wrappers")
    if args.num_examples > -1: 
        all_anns['annotations'] = all_anns['annotations'][0: args.num_examples]
        
    arg_wrappers = [ArgumentWrapper(annotation, question_lookup, glove_lookup, kmeans_wrapper_dict, args.penalty_factor, args.reduction) for annotation in all_anns['annotations']]

    examples_by_kmeans_score = []

    print(f"Delegating {len(arg_wrappers)} to {args.num_workers} jobs...")
    with get_context("spawn").Pool(args.num_workers) as p:
        outcome = p.map(make_training_example, arg_wrappers)

    out_path = Path(args.out_path)
    if "val" in args.ann_path:
        split="val"
    else:
        split="train"
    json_out_path = out_path.joinpath(f"{split}_kmeans_penalty_{args.penalty_factor}_{args.reduction}.json")
    with open(json_out_path, "w") as f1:
         json.dump(outcome, f1, indent=4)

    end_time = time.time()
    print(f"wrote to {json_out_path}")
    print(f"took {end_time - start_time:.2f}")