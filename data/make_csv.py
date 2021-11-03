import json 
from matplotlib import pyplot as plt 
import pathlib
import subprocess 
import csv 
import argparse 

def main(args):
    base_image_url = "https://cs.jhu.edu/~esteng/images_for_hit/" 
    image_root = pathlib.Path("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val2014")
    with open('/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val_kmeans_penalty_100.0_max.json') as f1:
        data = json.load(f1)

    filtered_data = [x for x in data if x['num_clusters'] == 2 and len(x['annotation_set']) > 5] 
    sorted_data = sorted(filtered_data, key = lambda x: x['difference_penalty'])[0: args.n]
    to_write = []
    for i in range(len(sorted_data)):
        small_data = sorted_data[i]
        question = small_data['question']['question']
        answers = ', '.join(small_data['annotation_set'])
        fname = get_image_fname(small_data['image_id'])
        image_path = image_root.joinpath(fname) 
        
        # copy image 
        subprocess.Popen(["cp", str(image_path), "to_copy"]) 

        line_to_write = {"image_url": base_image_url + fname, "question": question, "answers": answers}
        to_write.append(line_to_write)
    with open(args.out_path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames = ["image_url", "question", "answers"])
        writer.writeheader() 
        for row in to_write:
            writer.writerow(row)

def get_image_fname(num):
    num_digits = len("000000264110")
    n = len(str(num))
    n_zeros = num_digits-n
    zeros = "".join(['0'] * n_zeros)
    name = f"COCO_val2014_{zeros}{num}.jpg"
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out-path", type=str, default="hit/val.csv")
    args = parser.parse_args()
    main(args)
