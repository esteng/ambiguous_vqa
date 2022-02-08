local image_dir = "";
local cache_dir = "";
local transformer_model = "bert-base-cased";
local line_limit = "";
local cuda_device = -1;
local max_vocab_size = 500;

{
    dataset_reader: 
    {
        type: "vqav2",
        image_dir: image_dir,
        image_loader: 
            {type: "detectron"},
        image_featurizer: "resnet_backbone",
        region_detector: "faster_rcnn",
        answer_vocab: 
            {
                non_padded_namespaces: [],
                min_count: 
                    {source_tokens: 1},
                max_vocab_size: 
                    {source_tokens: max_vocab_size},
            },
        feature_cache_dir: cache_dir,
        tokenizer: 
            {
                type:"pretrained_transformer",
                model_name: transformer_model,
            },
        token_indexers: 
            {
                source_tokens: {
                    type: "pretrained_transformer",
                    model_name: transformer_model,
                },
            },
        cuda_device: cuda_device,
        max_instances: line_limit,
        multiple_answers_per_question: true,
        write_to_cache: true
    },
    model: 
    {

    },
    trainer: 
    {

    },
}