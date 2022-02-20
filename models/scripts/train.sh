#!/bin/bash

function train(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    rm -rf ${CHECKPOINT_DIR}/ckpt
    echo "Training a new transductive model for VQA..."
    python -um allennlp train \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    -s ${CHECKPOINT_DIR}/ckpt \
    ${TRAINING_CONFIG}
}

function eval(){
    export ALLENNLP_CACHE_ROOT="/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/"
    export TEST_DATA="unittest"
    echo "Evaluate a new transductive model for VQA at ${CHECKPOINT_DIR}..."
    python -um allennlp evaluate \
    --include-package allennlp.data.dataset_readers \
    --include-package allennlp.training \
    ${CHECKPOINT_DIR}/ckpt/model.tar.gz \
    ${TEST_DATA} 
}

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
    elif [[ "${action}" == "eval" ]]; then
        eval
    fi 
}


main "$@"
    
