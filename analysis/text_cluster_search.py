import argparse
import re
import pdb 
import pickle as pkl 
from collections import defaultdict
import json 
import pathlib 
import sys

import numpy as np 
from tqdm import tqdm
import torch 
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy 

from string_metrics import BertSimilarityScore, BleuSimilarityScore
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
logging.set_verbosity(50)



curr_path = pathlib.Path('').resolve().parent
sys.path.insert(0, str(curr_path.joinpath("hit3.0").joinpath("results")))
from process_csv import f1_score

np.set_printoptions(precision=2)


def preprocess(cluster_data):
    if type(cluster_data) in [dict, defaultdict]:
        # dealing with predicted clusters or preprocessed clusters
        return cluster_data.values()
    return cluster_data

def get_scores(clusters_by_qid_a, clusters_by_qid_b):
    scores = []
    for qid in clusters_by_qid_a.keys():
        cluster_a = preprocess(clusters_by_qid_a[qid])
        cluster_b = preprocess(clusters_by_qid_b[qid])
        f1_tuple = f1_score(cluster_a, cluster_b)
        f1_tuple = f1_tuple[0:-1]
        scores.append(f1_tuple)
    # print(scores)
    scores = np.array(scores)
    return np.mean(scores, axis=0)

# get the clusters from annotations 
def get_annotator_clusters(questions, annotations): 
    anns_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):

        qid, i = quest['question_id'].split("_")
        anns_by_qid[qid].append((quest, ann))

    clusters_by_qid = {}
    for qid, list_of_qas in anns_by_qid.items():
        clusters = defaultdict(list)
        for quest, ann in list_of_qas:
            rewritten = quest['new_question']
            answer = ann['answers'][0]['answer']
            answer_id = ann['answers'][0]['mturk_id']
            cluster_dict = {"answer": answer, "id": answer_id} 
            clusters[rewritten].append(cluster_dict)
        clusters_by_qid[qid] = clusters
    return clusters_by_qid

# get the clusters from kmeans preprocessing
def get_preprocessed_clusters(questions, annotations): 
    anns_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):

        qid, i = quest['question_id'].split("_")
        anns_by_qid[qid].append((quest, ann))

    clusters_by_qid = {}
    for qid, list_of_qas in anns_by_qid.items():
        clusters = defaultdict(list)
        for quest, ann in list_of_qas:
            answer = ann['answers'][0]['answer']
            answer_id = ann['answers'][0]['mturk_id']
            id_key, answer_id_suffix = answer_id.split(".")
            cluster_dict = {"answer": answer, "id": answer_id} 
            clusters[id_key].append(cluster_dict)
        clusters_by_qid[qid] = clusters
    return clusters_by_qid


def read_generations(output_path):
    flat_data_by_qid = {}
    if output_path.endswith(".json"):
        data = json.load(open(output_path))['questions']
        for qa_data in data:
            flat_data_by_qid[qa_data['question_id']] = qa_data['question']
    else:
        data = open(output_path).readlines()
        for line in data:
            batch_data = json.loads(line)
            for qid, generation in zip(batch_data['question_id'], batch_data['speaker_utterances'][0]):
                flat_data_by_qid[qid] = generation
    return flat_data_by_qid

def clean_text(text): 
    text = re.sub("<.*?>", "", text)
    text = text.strip() 
    return text 

def normalize(scores):
    # make other min zero 
    scores_no_zero = scores[scores > 0]
    min_score = np.min(scores_no_zero)
    # so everything becomes zero
    scores[scores == 0] += min_score
    # normalize scores so the min is zero and max is 1
    max_score = np.max(scores)
    if max_score == min_score:
        denom = max_score
    else:
        denom = max_score - min_score
    return (scores-min_score) / denom

def get_ann_clusters(questions, annotations):
    quests_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):
        
        qid, i = quest['question_id'].split("_")
        quests_by_qid[qid].append(quest['new_question'])
    return {k: set(v) for k, v in quests_by_qid.items()} 


def get_agglomerative_clusters(predictions_jsonl,
                            questions, 
                            annotations, 
                            embedder, 
                            num_clusters = None,
                            distance_threshold=0.5,
                            linkage = "ward",
                            kmeans = False): 
    generations_by_qid = read_generations(predictions_jsonl)
    anns_by_qid = defaultdict(list)
    answers_by_qid = defaultdict(list)
    missing = []
    for quest, ann in zip(questions, annotations):
        qid, i = quest['question_id'].split("_")
        try:
            generation = clean_text(generations_by_qid[quest['question_id']])
        except KeyError:
            missing.append(quest['question_id'])
            continue
        anns_by_qid[qid].append((generation, quest['question_id'], ann['answers'][0]['answer']))
        answers_by_qid[qid].append(ann['answers'])

    print(f"missing: {len(missing)} of {len(questions)}")
    with open("missing.pkl", "wb") as f:
        pkl.dump(missing, f)

    scores_by_qid = {} 
    clusts_by_qid = {}
    # Get matrix of scores 
    answer_clusters = {}
    ann_clusters = get_ann_clusters(questions, annotations)
    for qid, quest_list in tqdm(anns_by_qid.items()): 
        scores = np.zeros((len(quest_list), len(quest_list))) 
        done = []
        feats = embedder.encode(quest_list)
        if num_clusters is None and not kmeans: 
            model = AgglomerativeClustering(n_clusters=None, linkage=linkage, distance_threshold=distance_threshold)
        elif kmeans: 
            # get number of annotator clusters 
            num_clusters = len(ann_clusters[qid])
            model = KMeans(n_clusters=num_clusters)
        else:
            model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)  
        clust = model.fit_predict(feats) 
        

        clusts_by_qid[qid] = clust 
        answers_clustered = defaultdict(list)
        ans_list = answers_by_qid[qid]
        for i, idx in enumerate(clust):
            answer = ans_list[i]
            orig_id = answer[0]['mturk_id']
            cluster_dict = {"answer": answer[0]['answer'], "question": quest_list[i], "id": orig_id} 
            answers_clustered[f"g{idx}"].append(cluster_dict)
        answer_clusters[qid] = answers_clustered

    return answer_clusters


# get the clusters from predictions 
def get_prediction_clusters(predictions_jsonl,
                            questions, 
                            annotations, 
                            score_cls, 
                            t = 1.06, 
                            criterion = "centroid", 
                            method = "distance"):
    generations_by_qid = read_generations(predictions_jsonl)
    anns_by_qid = defaultdict(list)
    answers_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):
        qid, i = quest['question_id'].split("_")
        generation = clean_text(generations_by_qid[quest['question_id']])
        anns_by_qid[qid].append(generation)
        answers_by_qid[qid].append(ann['answers'])

    scores_by_qid = {} 
    clusts_by_qid = {}
    # Get matrix of scores 
    answer_clusters = {}
    for qid, quest_list in tqdm(anns_by_qid.items()): 
        scores = np.zeros((len(quest_list), len(quest_list))) 
        done = []
        for i, q1 in enumerate(quest_list): 
            for j, q2 in enumerate(quest_list):
                if i == j: 
                    scores[i,j] = 0.0 
                    continue
                sim_score = score_cls.get_similarity(q1, q2) 
                # print(q1, q2)
                # print(f"score: {sim_score}")
                if type(sim_score) == list:
                    # take the first element? 
                    sim_score = sim_score[0]
                
                scores[i,j] = 1/sim_score
                scores[j,i] = 1/sim_score
                # scores[i,j] = 1 - sim_score
                # scores[j,i] = 1 - sim_score
                done.append((i,j))
                done.append((j,i))
        
        # try normalizing 
        scores = normalize(scores)
        scores_by_qid[qid] = scores 
        scores = scipy.spatial.distance.squareform(scores)
        link = linkage(scores, method=method, metric="cosine")
        clust = fcluster(link, t=t, criterion=criterion)

        clusts_by_qid[qid] = clust 
        answers_clustered = defaultdict(list)
        ans_list = answers_by_qid[qid]
        for i, idx in enumerate(clust):
            answer = ans_list[i]
            orig_id = answer[0]['mturk_id']
            cluster_dict = {"answer": answer[0]['answer'], "question": quest_list[i], "id": orig_id} 
            answers_clustered[f"g{idx}"].append(cluster_dict)
        answer_clusters[qid] = answers_clustered

    return answer_clusters

class FileEmbedder:
    def __init__(self, checkpoint_dir, strategy):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.strategy = strategy 

    def encode(self, quest_list): 
        to_ret = []
        for quest, qid, answer in quest_list:
            checkpoint_path = self.checkpoint_dir.joinpath(f"{qid}.pt")
            encoded = torch.load(checkpoint_path, map_location='cpu') 
            if self.strategy == "mean": 
                to_ret.append(torch.mean(encoded, dim=0).numpy().reshape(-1))
            elif self.strategy == "max":
                to_ret.append(torch.max(encoded, dim=0).values.numpy().reshape(-1))
        return to_ret 

class BERTEmbedder: 
    def __init__(self, bert_model, strategy):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.strategy = strategy

    def encode(self, quest_list):
        to_ret = []
        for quest, qid, answer in quest_list:
            answer_tokenized = self.tokenizer(answer, return_tensors="pt")
            with torch.no_grad():
                answer_encoded = self.model(**answer_tokenized)['last_hidden_state'][0]
            # pdb.set_trace()
            if self.strategy == "mean": 
                to_ret.append(torch.mean(answer_encoded, dim=0).numpy().reshape(-1))
            elif self.strategy == "max":
                to_ret.append(torch.max(answer_encoded, dim=0).values.numpy().reshape(-1))
        return to_ret 


# TODO: elias: add just frozen VILT as a baseline instead of BERT, it's a better baseline than BERT 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-score", default="bert")
    parser.add_argument("--agglom", action="store_true")
    parser.add_argument("--kmeans", action="store_true")
    parser.add_argument("--hierarch", action="store_true")
    parser.add_argument("--criterion", default="distance")
    parser.add_argument("--method", default="ward")
    parser.add_argument("--t", type=float, default=1.13)
    parser.add_argument("--num-clusters", default=None, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--embedder", type=str, default="sentencebert", choices = ['sentencebert', 'path', 'bert'])
    parser.add_argument("--pooler", type=str, default="mean", choices = ['mean', 'max'])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="param_search/")
    parser.add_argument("--save-clusters", action="store_true")
    args = parser.parse_args() 

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.sim_score == "bert":
        score_cls = BertSimilarityScore(device="cuda:0")
    else:
        score_cls = BleuSimilarityScore() 

    if args.test:
        annotations = json.load(open("/home/estengel/annotator_uncertainty/hit3.0/results/cleaned_data/test/annotations.json"))['annotations']
        questions = json.load(open("/home/estengel/annotator_uncertainty/hit3.0/results/cleaned_data/test/questions.json"))['questions']
    else:
        annotations = json.load(open("/home/estengel/annotator_uncertainty/hit3.0/results/cleaned_data/dev/annotations.json"))['annotations']
        questions = json.load(open("/home/estengel/annotator_uncertainty/hit3.0/results/cleaned_data/dev/questions.json"))['questions']
    ann_clusters = get_annotator_clusters(questions, annotations)

    if args.test:
        pred_path = "/brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit/output/test_predictions_forced.jsonl" 
    else:
        pred_path = "/brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit/output/dev_predictions_forced.jsonl"
    
    data_to_write = []

    datapoint = {"sim_score": args.sim_score, "agglom": args.agglom, "kmeans": args.kmeans, "hierarch": args.hierarch, "criterion": args.criterion, "method": args.method, "t": args.t}
    if args.agglom or args.kmeans:

        if args.embedder == "sentencebert":

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        elif args.embedder == "bert": 
            embedder = BERTEmbedder("bert-base-uncased", args.pooler)
        else:
            embedder = FileEmbedder(args.checkpoint_dir, args.pooler)
            
        pred_clusters = get_agglomerative_clusters(pred_path,
                                                questions, 
                                                annotations,
                                                kmeans=args.kmeans,
                                                embedder = embedder,
                                                num_clusters = args.num_clusters,
                                                linkage=args.method,
                                                distance_threshold= args.t )

    else:
        pred_clusters = get_prediction_clusters(pred_path,
                                                questions, 
                                                annotations,
                                                # score_cls=score_cls,
                                                score_cls = BertSimilarityScore(device="cuda:0"),
                                                criterion=args.criterion,
                                                method= args.method,
                                                t=args.t) 

    if args.save_clusters:
        out_path = pathlib.Path(args.output_dir)
        with open(out_path.joinpath("pred_clusters.json"), "w") as f:
            json.dump(pred_clusters, f)
        with open(out_path.joinpath("ann_clusters.json"), "w") as f:
            json.dump(ann_clusters, f) 

    pred_to_ann = get_scores(pred_clusters, ann_clusters)
    datapoint['p'] = pred_to_ann[1]
    datapoint['r'] = pred_to_ann[2]
    datapoint['f1'] = pred_to_ann[0]

    out_path = pathlib.Path(args.output_dir).joinpath(f"{args.sim_score}_{args.agglom}_{args.kmeans}_{args.hierarch}_{args.criterion}_{args.method}_{args.t}_{args.num_clusters}_{args.embedder}_{args.pooler}.json")
    with open(out_path, "w") as f1:
        json.dump(datapoint, f1)
    # print(f"P: {pred_to_ann[1]*100:.2f}, R: {pred_to_ann[2]*100:.2f}, F1: {pred_to_ann[0]*100:.2f}")
