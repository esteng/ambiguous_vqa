import json
import shutil
import sys
from allennlp.commands import main
import argparse
 
parser = argparse.ArgumentParser('Train Network')
parser.add_argument('--config_file', type=str, default="allennlp/config/base/vilt/overfit_vilt_losses.jsonnet")
parser.add_argument('--serialization_dir', type=str, default="outputs/allennlp/ckpt_aml")
parser.add_argument('--debug', type=int, default=0, help='is debug')
args = parser.parse_args()

shutil.rmtree(args.serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "-um"
    "allennlp",  # command name, not used by main
    "train",
    "-s", args.serialization_dir,
    "--include-package", "allennlp.data.dataset_readers",
    "--include-package", "allennlp.training",
    args.config_file
]

main()
