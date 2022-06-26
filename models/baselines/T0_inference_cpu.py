#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 T0_inference.py

# Reference: https://github.com/huggingface/transformers/issues/16616

# Imports
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import json 
# To avoid warnings about parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from argparse import ArgumentParser


#################
# DeepSpeed Config
#################
def generate_ds_config(args):
    """
    ds_config notes

    - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
    faster.

    - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
    all official t5 models are bf16-pretrained

    - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
    - want CPU offload

    - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
    - which params should remain on gpus - the larger the value the smaller the offload size

    For indepth info on Deepspeed config see
    https://huggingface.co/docs/transformers/main/main_classes/deepspeed
    keeping the same format as json for consistency, except it uses lower case for true/false
    fmt: off
    """

    config = AutoConfig.from_pretrained(args.model_name)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    model_hidden_size = config.d_model

    # batch size has to be divisible by world_size, but can be bigger than world_size
    train_batch_size = args.batch_size * world_size

    config = {
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True 
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": args.offload,
                "nvme_path": args.nvme_offload_path,
                "pin_memory": True,
                "buffer_count": 6,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "aio": {
                "block_size": 262144,
                "queue_depth": 32,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_param_persistence_threshold": 10 * model_hidden_size
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }
    return config


#################
# Helper Methods
#################
def parse_args():
    """Parse program options"""
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="bigscience/T0pp", help="Name of model to load.")
    parser.add_argument("--batch-size", default=1, help="Effective batch size is batch-size * # GPUs")
    parser.add_argument("--one-shot", action='store_true')
    return parser.parse_args()


args = parse_args()

# Special version of T0
revision = None
if args.model_name in ["bigscience/T0", "bigscience/T0pp"]:
    revision = "sharded"
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, revision=revision).to("cpu")



# data_path = "dev.jsonl"
data_path = "/home/estengel/annotator_uncertainty/hit3.0/results/json_data/dev.jsonl"
with open(data_path) as f1:
    data = [json.loads(line) for line in f1]
    
prompt = "Rephrase the following question to match the answer.\nQuestion: {question}\nAnswer: {answer}\nNew Question: "
one_shot_prompt = "Rephrase the following question to match the answer.\nQuestion: {question0}\nAnswer: {answer0}\nNew Question: {new_question0}\n\nQuestion: {question1}\nAnswer: {answer1}\nNew Question: "
zero_shot_prompts = []
for example in data:
    old_question = example['original_question']
    for annotation in example["annotations"]:
        clusters_and_questions = zip(annotation['new_clusters'], annotation['new_questions'])
        for c, q in clusters_and_questions:
            for answer_dict in c:
                answer = answer_dict['content']
                filled_prompt = prompt.format(question=old_question, answer=answer)
                zero_shot_prompts.append(filled_prompt)

one_shot_prompts = []
for i in range(1, len(data)):
    previous_example = data[i-1]
    current_example = data[i]
    old_question = current_example['original_question']
    previous_question = previous_example['original_question']
    prev_ann = previous_example['annotations'][0]
    previous_answer = prev_ann['new_clusters'][0][0]['content']
    previous_new_question = prev_ann['new_questions'][0]
    for annotation in current_example["annotations"]:
        clusters_and_questions = zip(annotation['new_clusters'], annotation['new_questions'])
        for c, q in clusters_and_questions:
            for answer_dict in c:
                answer = answer_dict['content']
                filled_prompt = one_shot_prompt.format(question0=previous_question, answer0=previous_answer, new_question0 = previous_new_question, question1=old_question, answer1=answer)
                one_shot_prompts.append(filled_prompt)


# chunk_size = 1
# chunked_list = [all_prompts[i:i+chunk_size] for i in range(0, len(all_prompts), chunk_size)]
if args.one_shot:
    prompts = one_shot_prompts
else:
    prompts = zero_shot_prompts
# current_chuncked_lists = chunked_list[0]
for text_in in  prompts:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = tokenizer.encode(text_in, return_tensors="pt").to(device="cpu")

    # synced_gpus (bool, optional, defaults to False) —
    # Whether to continue running the while loop until max_length (needed for ZeRO stage 3) model_kwargs —
    # Additional model specific keyword arguments will be forwarded to the forward function of the model.
    # If model is an encoder-decoder model the kwargs should include encoder_outputs.
    with torch.no_grad():
        outputs = model.generate(inputs, do_sample=True, max_length=50, temperature=0.0001)
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"cpu:\n   in={text_in}\n  out={text_out}\n")
