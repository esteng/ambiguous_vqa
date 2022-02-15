local image_dir = "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/debug";
local data_dir = ""
local cache_dir = "";
local transformer_model = "bert-base-cased";
local line_limit = 10;
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
        multiple_answers_per_question: true
    },
    train_data_path: [data_dir + "balanced_real_val[0:10]", data_dir + "balanced_real_val[0:10]"]
    validation_data_path: [data_dir + "balanced_real_val[0:10]"]
    model: 
    {

    },
    data_loader:
    {
       type: "multiprocess",
       shuffle: true, 
    }
    trainer: 
    {
        optimizer: 
        {
            type: "adam",
            lr: 4e-5,
            weight_decay: 0.01
        },
        learning_rate_scheduler:
        {
            type: "linear_with_warmup",
            warmup_steps: 10000
        },
        validation_metric: ["vqa_acc"],
        patience: 5,
        num_epochs: 100,

    },
    random_seed: 12,
    numpy_seed: 12,
    pytorch_seed: 12,
}