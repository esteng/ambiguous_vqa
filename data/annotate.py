import json 
from matplotlib import pyplot as plt 
import pathlib
import subprocess 
image_root = pathlib.Path("/srv/local2/estengel/annotator_uncertainty/vqa/val2014")

with open('/srv/local2/estengel/annotator_uncertainty/vqa/val_kmeans_penalty_100.0_max.json') as f1:
    data = json.load(f1)

filtered_data = [x for x in data if x['num_clusters'] == 2 and len(x['annotation_set']) > 5] 
sorted_data = sorted(filtered_data, key = lambda x: x['difference_penalty'])

def get_image_fname(num):
    num_digits = len("000000264110")
    n = len(str(num))
    n_zeros = num_digits-n
    zeros = "".join(['0'] * n_zeros)
    name = f"COCO_val2014_{zeros}{num}.jpg"
    return name

def show_image(path):
    #p = subprocess.Popen(["/home/estengel/imgcat.sh", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.Popen(["/home/estengel/imgcat.sh", str(path)])
    #out, err = p.communicate() 
    #print(f"reading {image_path}") 
    #image = plt.imread(str(image_path))
    #plt.imshow(image)
    #plt.show()

    
with open("examples.json","w") as f1:
    for i in range(300):
        small_data = sorted_data[i]
        print(f"question: {small_data['question']['question']}")
        print(f"answers: {', '.join(small_data['annotation_set'])}")
        image_path = image_root.joinpath(get_image_fname(small_data['image_id']))
        #image = plt.imread(str(image_path))
        #plt.imshow(image)
        #plt.show()
        show_image(image_path) 
        code = input("keep? [y/N]")
        if code == "y": 
            f1.write(json.dumps(small_data, indent=4) + "\n")
        else:
            pass 
