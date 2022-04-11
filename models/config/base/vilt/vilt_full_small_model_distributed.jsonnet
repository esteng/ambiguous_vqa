local model_name = "bert-base-uncased";
local other_model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local effective_batch_size = 256;
local gpu_batch_size = 256;
local num_gpus = 2;
local line_limit = 10000;
// local line_limit = 16;

local construct_vocab = false;
// local dataset = "unittest";
local dataset = "balanced_real";
// local dataset = "unittest";
local dataset_vocab = "balanced_real";

local pooled_output_dim = 768; 

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": model_name};

local pretrained_token_indexers = {"tokens": {"type": "pretrained_transformer",
                                    "model_name": model_name,
                                  }
                        };
local token_indexers = {"tokens": {"type": "single_id",
                                    "namespace": "target_tokens"}};

{
  "dataset_reader": {
    "type": "vqav2",
    "image_dir": std.format("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s", dataset),
    // "tokenizer": tokenizer,
    "source_token_indexers": pretrained_token_indexers,
    "target_token_indexers": token_indexers,
    "max_instances": line_limit,
    "image_processing_batch_size": 16,
    "run_image_feature_extraction": false,
    "pass_raw_image_paths": true,
    "multiple_answers_per_question": false,
    "is_training": true,
    "is_validation": false,
    "use_precompute": false,
  },
  "validation_dataset_reader": {
    "type": "vqav2",
    "image_dir": std.format("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s", dataset),
    "tokenizer": tokenizer,
    "source_token_indexers": pretrained_token_indexers,
    "target_token_indexers": token_indexers,
    "max_instances": line_limit,
    "is_training": false,
    "is_validation": true,
    "image_processing_batch_size": 16,
    "multiple_answers_per_question": false,
    "run_image_feature_extraction": false,
    "pass_raw_image_paths": true,
    "use_precompute": false,
  },
//   "train_data_path": [std.format("%s", dataset)],
//   "validation_data_path": std.format("%s", dataset),
  "train_data_path": [std.format("%s_train", dataset)],
  "validation_data_path": std.format("%s_val[0:1000]", dataset),
  "model": {
    "type": "rsa_vqa",
    "label_namespace": "answers",
    "vision_language_encoder": {
      "type": "vilt", 
      "model_name": "dandelin/vilt-b32-finetuned-vqa",
      "half_precision": true,
    },
    "num_listener_steps": 1,
    "copy_speaker_listener": false,
    "pooled_output_dim": pooled_output_dim,
    "keep_tokens": false,
    "vqa_loss_factor": 10,
    "speaker_loss_factor": 1,
    "speaker_module": 
        {"type": "simple_speaker",
        "target_namespace": "target_tokens",
        "encoder_in_dim": pooled_output_dim,
        "encoder_num_layers": 2,
        "encoder_hidden_dim": 256,
        "encoder_dropout": 0.2,
        "encoder_activation": "relu",
                      "decoder": {
                        "type": "auto_regressive_seq_decoder",
                        "decoder_net": 
                            {"type": "stacked_self_attention",
                              "decoding_dim": 256,
                              "target_embedding_dim": 256,
                              "feedforward_hidden_dim": 512,
                              "num_layers": 2,
                              "num_attention_heads": 2,
                              "dropout_prob": 0.2,
                              "residual_dropout_prob": 0.2,
                              "attention_dropout_prob": 0.2
                            },
                          "max_decoding_steps": 50,
                          "target_namespace": "target_tokens",
                          "target_embedder": {
                              "vocab_namespace": "target_tokens",
                              "embedding_dim": 256
                            },
                          "scheduled_sampling_ratio": 0.5,
                          "beam_size": 5,
                        },
                       "dropout": 0.2},
    "listener_module": {"type": "simple_listener",
        "encoder_in_dim": 256,
        "encoder_num_layers": 2,
        "encoder_hidden_dim": 256,
        "encoder_dropout": 0.2,
        "encoder_activation": "relu"
                        },
    
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    //[if !construct_vocab then "max_instances_in_memory"]: 10240
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    // "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "type": "warmup_gradient_descent",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 4e-3}]
      ],
    },
    // "learning_rate_scheduler": {
    //   "type": "linear_with_warmup",
    //   //"num_steps_per_epoch": std.ceil(0 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
    //   "warmup_steps": 4000
    // },
    "validation_metric": "+vqa_score",
    "save_warmup": 0,
    "patience": 300,
    "num_epochs": 200,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size, 
  },
  "random_seed": 12,
  "numpy_seed": 12,
  "pytorch_seed": 12,
}
