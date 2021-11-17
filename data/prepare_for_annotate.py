import json 
from matplotlib import pyplot as plt 
import pathlib
import subprocess 
import argparse 
import csv 
from tqdm import tqdm 

def get_image_fname(num):
    num_digits = len("000000264110")
    n = len(str(num))
    n_zeros = num_digits-n
    zeros = "".join(['0'] * n_zeros)
    name = f"COCO_val2014_{zeros}{num}.jpg"
    return name

def main(args):
    image_root = pathlib.Path(args.image_path)
    
    print(f"Loading data from {args.kmeans_path}") 
    with open(args.kmeans_path) as f1:
        data = json.load(f1)

    filtered_data = [x for x in data if x['num_clusters'] == 2 and len(x['annotation_set']) > 5] 
    sorted_data = sorted(filtered_data, key = lambda x: x['difference_penalty'])

    to_copy_path = pathlib.Path("to_copy")
    to_copy_path.mkdir(exist_ok=True)

    base_url = "http://cs.jhu.edu/~esteng/images_for_hit/"

    all_lines = []
    print(f"Taking top {args.n} examples") 
    for example in tqdm(sorted_data[0:args.n]):
        image_id = example['image_id']
        fname = get_image_fname(image_id) 
        path_to_image = image_root.joinpath(fname) 
       
        # copy 
        subprocess.Popen(["cp", str(path_to_image), "to_copy"])  

        url = f"{base_url}/{fname}"
        question = example['question']['question']    
        answers = ", ".join(example['annotation_set']) 
        csv_line = {"image_url": url, "question": question, "answers": answers}
        all_lines.append(csv_line) 

    print(f"Writing to {args.out_path}") 
    with open(args.out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames = ["image_url", "question", "answers"]) 
        writer.writeheader()
        writer.writerows(all_lines) 
           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val2014")
    parser.add_argument("--kmeans-path", type=str, default="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val_kmeans_penalty_100.0_max.json") 
    parser.add_argument("--out-path", type=str, required = True) 
    parser.add_argument("--n", type=int, default=300) 
    args = parser.parse_args() 
    main(args) 
