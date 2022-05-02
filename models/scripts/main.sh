#!/bin/bash

function train(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    mkdir -p ${CHECKPOINT_DIR}/code
    echo "Copying code" 
    cp -r allennlp ${CHECKPOINT_DIR}/code
    cp -r allennlp_models ${CHECKPOINT_DIR}/code
    echo "Training a new transductive model for VQA..."
    python -um allennlp train \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    -s ${CHECKPOINT_DIR}/ckpt \
    ${TRAINING_CONFIG}
}

function precompute_intermediate(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    python -um allennlp evaluate \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} \
    --cuda-device 0 \
    --precompute-intermediate \
    --retrieval-save-dir ${SAVE_DIR}
}


function resume(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Training a new transductive model for VQA..."
    python -um allennlp train \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    --recover \
    -s ${CHECKPOINT_DIR}/ckpt \
    ${TRAINING_CONFIG}
}

function eval(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    python -um allennlp evaluate \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} \
    --cuda-device 0 \
    --predictions-output-file ${CHECKPOINT_DIR}/output/predictions.jsonl 
}

function predict(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    python -um allennlp evaluate \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} \
    --cuda-device 0 \
    --ignore-logits \
    --batch-size 5 \
    --predictions-output-file ${CHECKPOINT_DIR}/output/predictions.jsonl 
}

function min_gen(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    mkdir -p ${CHECKPOINT_DIR}/output
    python -um allennlp min_gen \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} \
    --cuda-device 0 \
    --predictions-output-file ${CHECKPOINT_DIR}/output/big_min_gen_debug_steps_100.jsonl \
    --descent-strategy steps \
    --num-descent-steps 100 \
    --lr 0.05
}
    # --descent-loss-threshold  \
    # --descent-strategy thresh \
    # --mix-ratio 0.5 \
    # --mix-strategy end \

function min_gen_save(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    mkdir -p ${CHECKPOINT_DIR}/output
    python -um allennlp min_gen \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} \
    --descent-strategy steps \
    --num-descent-steps 100 \
    --cuda-device 0 \
    --precompute-intermediate \
    --retrieval-save-dir ${SAVE_DIR} \
    --lr 0.05
}
    #--mix-strategy end \
    #--mix-ratio 0.5 \
    #--descent-loss-threshold 0.15 \
    #--descent-strategy thresh \

    #--mix-strategy continuous \
    #--mix-ratio 0.5 \
    #--descent-strategy steps \
    #--num-descent-steps 100 \

function usage() {

    echo -e 'usage: train.sh [-h] -a action'
    echo -e '  -a do [train|test|all].'
    echo -e "  -d checkpoint_dir (Default: ${CHECKPOINT_DIR})."
    echo -e "  -c training_config (Default: ${TRAINING_CONFIG})."
    echo -e "  -i test_data (Default: ${TEST_DATA})."
    echo -e 'optional arguments:'
    echo -e '  -h \t\t\tShow this help message and exit.'

    exit $1

}


function parse_arguments() {

    while getopts ":h:a:d:c:i:" OPTION
    do
        case ${OPTION} in
            h)
                usage 1
                ;;
            a)
                action=${OPTARG:='train'}
                ;;
            d)
                CHECKPOINT_DIR=${OPTARG:=${CHECKPOINT_DIR}}
                ;;
            c)
                TRAINING_CONFIG=${OPTARG:=${TRAINING_CONFIG}}
                ;;
            i)
                TEST_DATA=${OPTARG:=${TEST_DATA}}
                ;;
            ?)
                usage 1
                ;;
        esac
    done

    if [[ -z ${action} ]]; then
        echo ">> Action not provided"
        usage
        exit 1
    fi
}


function main() {

    parse_arguments "$@"
    if [[ "${action}" == "test" ]]; then
        test
    elif [[ "${action}" == "train" ]]; then
        train
    elif [[ "${action}" == "resume" ]]; then
        resume
    elif [[ "${action}" == "precompute" ]]; then
        precompute_intermediate
    elif [[ "${action}" == "eval" ]]; then
        eval
    elif [[ "${action}" == "predict" ]]; then
        predict 
    elif [[ "${action}" == "min_gen" ]]; then
        min_gen 
    elif [[ "${action}" == "min_gen_save" ]]; then
        min_gen_save 
    fi 
}


main "$@"
    
