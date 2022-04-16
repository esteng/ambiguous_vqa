import argparse 
import torch 
import numpy as np 
import pathlib
import json 
from tqdm import tqdm 

def load_questions_and_annotations(path):
    path = pathlib.Path(path) 
    question_path = path.joinpath("questions.json")
    annotation_path = path.joinpath("annotations.json")
    print(f"loading from {path}")
    with open(question_path) as qf, open(annotation_path) as af:
        questions = json.load(qf)
        annotations = json.load(af)
    return questions, annotations

def load_training_vectors(path):
    path = pathlib.Path(path)
    vectors = path.glob("*.pt")
    return list(vectors) 

def lookup_question_and_annotation(questions, annotations, question_id):
    for i, (quest, ann) in enumerate(zip(questions['questions'], annotations['annotations'])): 
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
    parser.add_argument("--make-shards", action="store_true") 
    parser.add_argument("--shard-size", type=int, default=20000)
    args = parser.parse_args() 

    # load questions and annotations
    questions, annotations = load_questions_and_annotations(args.train_path)
    # load training vectors
    train_vector_paths = [x for x in load_training_vectors(args.precompute_path)]

    shard_out_path = pathlib.Path(args.precompute_path).joinpath("shards_filtered")
    if args.make_shards:
        # make shards
        make_shards(train_vector_paths, shard_out_path, args.shard_size)

    else:
        # load shards
        train_matrix = load_shards(shard_out_path)

        # load test vectors
        test_vector_paths = load_training_vectors(args.test_path)
        test_vectors = [torch.load(x, map_location="cpu").reshape(1, -1) for x in tqdm(test_vector_paths)]
        test_matrix = torch.cat(test_vectors, dim=0)

        # compute similarities
        similarities = sim_matrix(test_matrix, train_matrix)
        similarities = similarities.cpu().numpy()

        # save similarities
        similarities_path = pathlib.Path(args.precompute_path).joinpath("similarities.npy")
        np.save(similarities_path, similarities)

