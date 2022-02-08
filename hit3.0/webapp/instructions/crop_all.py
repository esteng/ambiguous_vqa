import subprocess
import pathlib
import re
#  convert -crop -0-125 Screen\ Shot\ 2022-02-07\ at\ 10.44.24\ AM\ \(2\).png out.png


path = pathlib.Path("./")
all_dirs = path.glob("*") 

for d in all_dirs: 
    all_files = d.glob("*.png") 
    for i, fname in enumerate(all_files): 
        dir_name = fname.parent.stem
        out_name = f"{dir_name}_{i}.png"
        out_file = fname.parent.joinpath(out_name) 
        print(f"convert {fname} {out_file}") 
        #fname = re.sub(" ", "\ ", str(fname))
        #fname = re.sub("\(","\\(", fname)
        #fname = re.sub("\)","\\)", fname)
        subprocess.Popen(["convert", "-crop", "-0-155", fname, out_file])
