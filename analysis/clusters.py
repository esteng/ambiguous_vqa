import numpy as np 
import torch 
from collections import defaultdict
import json 
import pathlib 
import sys
import argparse
from pathlib import Path 

curr_path = Path('').resolve().parent
sys.path.insert(0, str(curr_path.joinpath("hit3.0").joinpath("results")))
from process_csv import f1_score

from sklearn.cluster import OPTICS, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from tqdm import tqdm 





class BayesianGMMCluster:
    def fit(self, vectors):
        gmm_wrapper = BayesianGaussianMixture(n_components = vectors.shape[0])
        gmm_wrapper.fit(vectors)
        weights = np.round(gmm_wrapper.weights_, 1)
        probs = gmm_wrapper.predict_proba(vectors)
        non_zero_weights = weights > 0
        non_zero_weights = non_zero_weights.reshape(-1, 1)
        non_zero_weights = np.tile(non_zero_weights, reps=probs.shape[1])
        probs *= non_zero_weights
        
        labels = np.argmax(probs, axis=1)
        # print(scores_data)
        # score_array = np.array(scores_data)
        # min_row = np.argmin(score_array)
        self.labels_ = labels
        # sys.exit() 

class GMMCluster:
    def __init__(self, use_aic=True):
        self.gmm_wrapper_dict = {k: GaussianMixture(n_components=k, random_state=12, n_init=2) for k in range(0, 10)}
        self.use_aic = use_aic

    def fit(self, vectors):
        scores_data = []
        assignments = []
        all_labels = []
        for k in range(1, vectors.shape[0]):
            gmm_wrapper = self.gmm_wrapper_dict[k]    
            gmm_wrapper.fit(vectors) 
            if self.use_aic:
                aic = gmm_wrapper.aic(vectors)
                n_params = gmm_wrapper._n_parameters()
                if vectors.shape[0] - n_params - 1 == 0:
                    aicc = np.inf
                else:

                    aicc = (2 * n_params**2 + 2*n_params)/(vectors.shape[0] - n_params - 1) + aic
                scores_data.append(aicc)
            else:
                scores_data.append(gmm_wrapper.bic(vectors))

            labels = gmm_wrapper.predict(vectors)
            all_labels.append(labels)
        score_array = np.array(scores_data)
        min_row = np.argmin(score_array)
        self.labels_ = all_labels[min_row]


class KMeansCluster:
    def __init__(self, penalty_factor=3.0):
        self.kmeans_wrapper_dict = {k: KMeans(n_clusters=k, random_state=12) for k in range(0, 10)}
        self.penalty_factor = penalty_factor
        self.labels_ = None

    def fit(self, vectors):
        scores_data = []
        assignments = []
        all_labels = []
        for k in range(2, vectors.shape[0]):
            kmeans_wrapper = self.kmeans_wrapper_dict[k]    
            # run kmeans
            kmeans = kmeans_wrapper.fit(vectors) 
            all_labels.append(kmeans.labels_)
            centers = kmeans.predict(vectors) 
            num_centers = len(set(centers))
            # intertia is sum of squared distances of samples to their closest cluster center 
            inertia = kmeans.inertia_
            # penalty depends on how many centers you used compared to how many examples you have 
            penalty = self.penalty_factor * (num_centers-1)/ (vectors.shape[0]-1) 
            # we want balanced clusters
            avg = int(vectors.shape[0] / num_centers)
            num_per_center = {c_name: sum([1 for c in centers if c == c_name]) for c_name in centers}
            diffs = [abs(avg - num_per_center[c]) for c in centers]
            difference_penalty = sum(diffs)
            score = inertia + penalty + difference_penalty 

            # score = kmeans.inertia_ + num_centers ** penalty_factor
            # high intertia means dispersed, low means fits well to the number of clusters
            scores_data.append((score, inertia, penalty, difference_penalty, num_centers))

            assignments.append(centers)
            # if you hit zero inertia, no sense in going further 
            if kmeans.inertia_ < 1e-16:
                break

        score_array = np.array(scores_data)
        min_row = np.argmin(score_array[:,0])
        self.labels_ = all_labels[min_row]


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



def cluster_vectors(vectors, cluster_method='optics', do_pca=False, pca_comp_max=2):
    vidxs = [x[1] for x in vectors]
    vectors = np.vstack([x[0] for x in vectors]).reshape(len(vectors), -1)
    if do_pca:
        from sklearn.decomposition import PCA
        comp = min(vectors.shape[0], pca_comp_max)
        pca = PCA(n_components=comp)
        vectors = pca.fit_transform(vectors)

    if cluster_method == "optics":
        clust = OPTICS(min_samples=2, metric='euclidean')
    elif cluster_method == "mean_shift":
        bw = estimate_bandwidth(vectors, quantile=0.5)
        if bw < 0.0001:
            bw = 0.0001
        clust = MeanShift(bandwidth=bw)
    elif cluster_method == "kmeans": 
        clust = KMeansCluster(penalty_factor=52.0)
    elif cluster_method == "gmm": 
        clust = GMMCluster(use_aic=True)
    elif cluster_method == "bayes_gmm": 
        clust = BayesianGMMCluster()

    else:
        raise AssertionError("Unknown cluster method")
    clust.fit(vectors)
    clusters = defaultdict(list)
    max_label = max(clust.labels_) + 1
    for i, vidx in enumerate(vidxs):
        label = clust.labels_[i]
        if label == -1:
            label = max_label
        clusters[label].append(vidx)
    return clusters 

# get the clusters from predictions 
def get_prediction_clusters(questions, annotations, save_dir, cluster_method='optics', do_pca=False, pca_comp_max=2):
    anns_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):
        qid, i = quest['question_id'].split("_")
        anns_by_qid[qid].append((quest, ann))

    vectors_by_qid = defaultdict(list)
    answers_by_qid = defaultdict(list)

    # print(anns_by_qid)
    for qid, list_of_qas in anns_by_qid.items():
        image_id = list_of_qas[0][0]['image_id']
        for quest, ann in list_of_qas:
            qid, i = quest['question_id'].split("_")
            path = pathlib.Path(save_dir).joinpath(f"{image_id}_{qid}_{i}_0.pt")
            vector = torch.load(path, map_location=torch.device('cpu')).detach().numpy()
            answer = ann['answers'][0]['answer']
            answer_id = ann['answers'][0]['mturk_id']
            vectors_by_qid[qid].append((vector, int(i)))
            answers_by_qid[qid].append({"answer": answer, "id": answer_id})

    clusters_by_qid = {}
    for qid, vectors in vectors_by_qid.items():
        clusters = cluster_vectors(vectors, cluster_method=cluster_method, do_pca=do_pca, pca_comp_max=pca_comp_max)

        clusters = {k: [answers_by_qid[qid][idx] for idx in v ] for k, v in clusters.items()}
        clusters_by_qid[qid] = clusters

    return clusters_by_qid



def preprocess(cluster_data):
    if type(cluster_data) in [dict, defaultdict]:
        # dealing with predicted clusters or preprocessed clusters
        return cluster_data.values()
    return cluster_data

def get_scores(clusters_by_qid_a, clusters_by_qid_b):
    f1_scores = []
    p_scores = []
    r_scores = []
    cluster_lens_a = []
    cluster_lens_b = []
    for qid in clusters_by_qid_a.keys():
        cluster_a = preprocess(clusters_by_qid_a[qid])
        cluster_b = preprocess(clusters_by_qid_b[qid])
        f1, p, r = f1_score(cluster_a, cluster_b)
        f1_scores.append(f1)
        p_scores.append(p)
        r_scores.append(r)
        cluster_lens_a.append(len(cluster_a))
        cluster_lens_b.append(len(cluster_b))
    return np.mean(f1_scores), np.mean(p_scores), np.mean(r_scores), np.mean(cluster_lens_a), np.mean(cluster_lens_b)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default = None, required=True)
    parser.add_argument("--data_dir", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/dev_from_mturk/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    annotations = json.load(open(data_dir.joinpath("annotations.json")))['annotations']
    questions = json.load(open(data_dir.joinpath("questions.json")))['questions']

    ann_clusters = get_annotator_clusters(questions, annotations)
    # print(json.dumps(ann_clusters, indent=4))

    glove_clusters = get_preprocessed_clusters(questions, annotations)
    # print(json.dumps(glove_clusters, indent=4))
    (f1_glove_to_ann, 
     p_glove_to_ann, 
     r_glove_to_ann, 
     all_cluster_lens_glove, 
     all_cluster_lens_ann) = get_scores(glove_clusters, ann_clusters)


    cluster_methods = ['bayes_gmm', 'gmm', 'optics', 'mean_shift']
    pca_dims = [2,3,4,5,6,7,8,9,10]
    methods = [[None for i in range(len(pca_dims))] for j in range(len(cluster_methods))]
    all_cluster_lens_pred = np.zeros((len(cluster_methods), len(pca_dims)))
    f1_scores = np.zeros((len(cluster_methods), len(pca_dims)))
    p_scores = np.zeros((len(cluster_methods), len(pca_dims)))
    r_scores = np.zeros((len(cluster_methods), len(pca_dims)))
    for i, method in tqdm(enumerate(cluster_methods), total=len(cluster_methods)):
        for j, pca_dim in enumerate(pca_dims):
            pred_clusters = get_prediction_clusters(questions, 
                                    annotations, 
                                    str(checkpoint_dir), 
                                    method, 
                                    do_pca=True,
                                    pca_comp_max=pca_dim)


            (f1_pred_to_ann, 
             p_pred_to_ann, 
             r_pred_to_ann,
             cluster_lens_pred, __) = get_scores(pred_clusters, ann_clusters)
            all_cluster_lens_pred[i,j] = cluster_lens_pred
            f1_scores[i, j] = f1_pred_to_ann
            p_scores[i, j] = p_pred_to_ann
            r_scores[i, j] = r_pred_to_ann
            methods[i][j] = (method, pca_dim)

    print(f"Avg number clusters in annotations: {all_cluster_lens_ann}")
    print(f"Avg number clusters in glove: {all_cluster_lens_glove}")
    print(f"scores from glove:")
    print(f"\tF1: {f1_glove_to_ann}, P: {p_glove_to_ann}, R: {r_glove_to_ann}")
    methods = np.array(methods)
    best_f1 = np.max(f1_scores)
    best_p = p_scores[f1_scores == best_f1]
    best_r = r_scores[f1_scores == best_f1]
    best_method = methods[f1_scores == best_f1][0]
    best_cluster_lens_pred = all_cluster_lens_pred[f1_scores == best_f1][0]
    print(f"Best method: {best_method}")
    print(f"Avg number clusters: {best_cluster_lens_pred}")
    print(f"Scores from clustering:")
    print(f"\tF1: {best_f1}, P: {best_p[0]}, R: {best_r[0]}")