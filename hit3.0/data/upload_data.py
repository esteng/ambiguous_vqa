import subprocess
import json
import sys
import pathlib 

data = json.load(open(sys.argv[1]))

def get_image_fname(num):
    num_digits = len("000000264110")
    n = len(str(num))
    n_zeros = num_digits-n
    zeros = "".join(['0'] * n_zeros)
    name = f"COCO_val2014_{zeros}{num}.jpg"
    return name

image_root = pathlib.Path("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/train2014/") 

to_copy_path = pathlib.Path("to_copy")
to_copy_path.mkdir(exist_ok=True)

for example in data:
    image_id = example['image']
    # fname = get_image_fname(image_id) 
    path_to_image = image_root.joinpath(image_id) 
   
    # copy 
    subprocess.Popen(["cp", str(path_to_image), "to_copy"])  


