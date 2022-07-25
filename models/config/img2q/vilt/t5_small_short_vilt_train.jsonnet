local model_name = "bert-base-uncased";
local other_model_name = "bert-base-uncased";
local gpu_batch_size = 60;
local num_gpus = 1;
local effective_batch_size = num_gpus * gpu_batch_size;
local line_limit = 10000;

local construct_vocab = false;
local dataset = "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/filtered";

local pooled_output_dim = 768; 

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": model_name};

local pretrained_token_indexers = {"tokens": {"type": "pretrained_transformer",
                                    "model_name": model_name,
                                  }
                        };
local token_indexers = {"tokens": {"type": "single_id",
                                    "namespace": "target_tokens"}};
local vilt_model ={
      type: 'vilt',
      model_name: '/brtx/605-nvme1/estengel/annotator_uncertainty/models/finetune_vilt_pytorch/',
      half_precision: false,
    };

{
  "dataset_reader": {
    "type": "vqa_for_img2quest",
    "image_dir": "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/balanced_real",
    "source_token_indexers": pretrained_token_indexers,
    "target_token_indexers": token_indexers,
    "max_instances": line_limit,
    "image_processing_batch_size": 16,
    "run_image_feature_extraction": false,
    "pass_raw_image_paths": true,
    "multiple_answers_per_question": true,
    "use_multilabel": true,
    "is_training": true,
    "is_validation": false,
    "use_precompute": false,
    "pretrained_model": vilt_model,
  },
  "validation_dataset_reader": {
    "type": "vqa_for_img2quest",
    "image_dir": "/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/balanced_real",
    "tokenizer": tokenizer,
    "source_token_indexers": pretrained_token_indexers,
    "target_token_indexers": token_indexers,
    "max_instances": line_limit,
    "is_training": false,
    "is_validation": true,
    "image_processing_batch_size": 16,
    "multiple_answers_per_question": true,
    "use_multilabel": true,
    "run_image_feature_extraction": false,
    "pass_raw_image_paths": true,
    "use_precompute": false,
    "pretrained_model": vilt_model,
  },
  "train_data_path": [std.format("%s", dataset)],
  // "validation_data_path": std.format("%s[0:1000]", val_dataset),
  // "train_data_path": [std.format("%s_train", dataset)],
  "validation_data_path": "balanced_real_val[0:2000]",
  "model": {
    "type": "t5_img_ans_2_question",
    "vision_language_encoder": vilt_model,
    "no_vilt_train": false,
    "pooled_output_dim": pooled_output_dim,
    "t5_model_name": "t5-small", 
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
  },
  "vocabulary": {
    "min_count": {
      "target_tokens": 10,
      "answers": 8,
    },
  },
  "datasets_for_vocab_creation": ["train"],
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    // "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    "type": "warmup_gradient_descent",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-4,
      "correct_bias": true,
      "weight_decay": 0.01,
      // "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        // [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 4e-3}]
      // ],
    },
    // "learning_rate_scheduler": {
    //   "type": "linear_with_warmup",
    //   //"num_steps_per_epoch": std.ceil(0 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
    //   "warmup_steps": 4000
    // },
    "validation_metric": "+BLEU",
    "save_warmup": 0,
    "patience": 5,
    "num_epochs": 200,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size, 
  },
  "random_seed": 12,
  "numpy_seed": 12,
  "pytorch_seed": 12,
}
