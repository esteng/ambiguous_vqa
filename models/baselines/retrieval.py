import argparse 
import torch 
import numpy as np 
import pathlib
import json 
import pdb 
from tqdm import tqdm 

def load_questions_and_annotations(path):
    path = pathlib.Path(path) 
    question_path = path.joinpath("questions.json")
    annotation_path = path.joinpath("annotations.json")
    print(f"loading from {path}")
    with open(question_path) as qf, open(annotation_path) as af:
        questions = json.load(qf)['questions']
        annotations = json.load(af)['annotations']
    return questions, annotations

def load_training_vectors(path):
    path = pathlib.Path(path)
    vectors = path.glob("*.pt")
    return list(vectors) 

def lookup_question_and_annotation(questions, annotations, question_id):
    for i, (quest, ann) in enumerate(zip(questions, annotations)): 
        if str(quest['question_id']) == str(question_id):
            return quest, ann
    return -1, -1 

def make_shards(paths, out_path, shard_size=20000): 
    print("making shards")
    for shard_start in range(0, len(paths), shard_size): 
        shard_end = shard_start + shard_size
        shard_paths = paths[shard_start:shard_end]
        print(f"shard {shard_start} to {shard_end}")
        shard_vecs = [torch.load(x, map_location="cpu").reshape(1, -1) for x in tqdm(shard_paths)]
        shard_matrix = torch.cat(shard_vecs, dim=0)
        shard_out_path = out_path.joinpath("shard_{}_{}.pt".format(shard_start, shard_end))
        torch.save(shard_matrix, shard_out_path)
            
def load_shards(path):
    shards = []
    for shard in path.glob("*.pt"):
        shards.append(torch.load(shard, map_location="cpu")) 
    return torch.cat(shards, dim=0) 

def remove_nonfiltered(questions, train_matrix, qid_to_idx):
    filtered_qid_to_idx = {}
    for q in questions:
        qid = str(q['question_id'])
        filtered_qid_to_idx[qid] = qid_to_idx[qid]

    num_examples = len(filtered_qid_to_idx)
    sim_dim = train_matrix.shape[1]

    matrix_to_keep = torch.zeros((num_examples, sim_dim))
    for i, (qid, idx) in enumerate(filtered_qid_to_idx.items()):
        matrix_to_keep[i, :] = train_matrix[idx, :]
    return matrix_to_keep 

def get_train_lookup(paths):
    index_to_qid = {}
    qid_to_index = {}
    for i, path in enumerate(paths):
        image_id, qid, layer = pathlib.Path(path).stem.split("_")
        index_to_qid[i] = qid
        qid_to_index[qid] = i
    return index_to_qid, qid_to_index 

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default = "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/filtered") 
    parser.add_argument("--test-path", type=str, default = "/home/estengel/annotator_uncertainty/models/test_fixtures/data/real_vqa")
    parser.add_argument("--precompute-path", type=str, default = "/brtx/602-nvme2/estengel/annotator_uncertainty/vqa/precomputed_retrieval/")
    parser.add_argument("--out-path", type=str, default="baselines/outputs/")
    parser.add_argument("--make-shards", action="store_true") 
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=20000)
    args = parser.parse_args() 

    # load questions and annotations
    questions, annotations = load_questions_and_annotations(args.train_path)
    test_questions, test_annotations = load_questions_and_annotations(args.test_path)
    # load training vectors
    train_vector_paths = [x for x in load_training_vectors(args.precompute_path)]

    shard_out_path = pathlib.Path(args.precompute_path).joinpath("shards_filtered")
    if args.make_shards:
        # make shards
        make_shards(train_vector_paths, shard_out_path, args.shard_size)

    else:
        # load shards
        train_matrix = load_shards(shard_out_path)
        print(train_matrix.shape)
        train_index_to_qid, train_qid_to_index = get_train_lookup(train_vector_paths)
        train_matrix = remove_nonfiltered(questions, train_matrix, train_qid_to_index)
        print(train_matrix.shape)

        # load test vectors
        test_vector_paths = load_training_vectors(pathlib.Path(args.test_path).joinpath("precomputed")) 
        test_vectors = [torch.load(x, map_location="cpu").reshape(1, -1) for x in tqdm(test_vector_paths)]
        test_matrix = torch.cat(test_vectors, dim=0)

        # compute similarities
        similarities = sim_matrix(test_matrix, train_matrix)
        similarities = similarities.detach().cpu().numpy()

        # save similarities
        similarities_path = pathlib.Path(args.out_path).joinpath("similarities.npy")
        np.save(similarities_path, similarities)

        # get top similarity indices 
        indices = np.argpartition(similarities, -args.top_k, axis=1)[:, -args.top_k:]
        results = []
        for example_idx in range(indices.shape[0]):
            source_question = test_questions[example_idx]
            source_annotation = test_annotations[example_idx]
            top_questions, top_annotations = [], []

            for idx in indices[example_idx]:
                question, annotation = questions[idx], annotations[idx]
                # qid = train_index_to_qid[idx]
                # question, annotation = lookup_question_and_annotation(questions, annotations, qid)
                top_questions.append(question)
                top_annotations.append(annotation) 

            results.append({"original_question": source_question,
                            "original_annotation": source_annotation,
                            "top_questions": top_questions,
                            "top_annotations": top_annotations})
        
        pdb.set_trace() 
        

