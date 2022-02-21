// local model_name = "bert-base-uncased";
local model_name = "prajjwal1/bert-tiny";
local other_model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;
local line_limit = 2;

local construct_vocab = false;
local dataset = "unittest";
local dataset_vocab = "balanced_real";

local pooled_output_dim = 32; 

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": model_name};

local token_indexers = {"tokens": {"type": "pretrained_transformer",
                                    "model_name": model_name
                                  }
                        };

  // local vocabulary = if construct_vocab then {
  //       // read the files to construct the vocab
//       "min_count": {"answers": 9}
//     } else {
//       // // read the constructed vocab
//       "type": "from_files",
//       "directory": std.format(
//         "https://storage.googleapis.com/allennlp-public-data/vqav2/vilbert_vqa_%s.%s.vocab.tar.gz",
//         [dataset_vocab, other_model_name])

//     };
// local vocabulary = {"type": "from_instances", "min_count": {"answers": 1}};

{
  "dataset_reader": {
    "type": "vqav2",
    "image_dir": std.format("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s", dataset),
    [if !construct_vocab then "feature_cache_dir"]: std.format("/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s/feature_cache", dataset),
    #"image_dir": std.format("/Users/dirkg/Documents/data/vision/vqa/%s", dataset),
    #[if !construct_vocab then "feature_cache_dir"]: std.format("/Users/dirkg/Documents/data/vision/vqa/%s/feature_cache", dataset),
    [if !construct_vocab then "image_loader"]: "detectron",
    [if !construct_vocab then "image_featurizer"]: "resnet_backbone",
    [if !construct_vocab then "region_detector"]: "faster_rcnn",
    "tokenizer": tokenizer,
    "token_indexers": token_indexers,
    "max_instances": line_limit,
    "image_processing_batch_size": 16,
    // "answer_vocab": if construct_vocab then null else vocabulary,
    // "multiple_answers_per_question": !construct_vocab,
    multiple_answers_per_question: false,

  },
  // "validation_dataset_reader": self.dataset_reader {
  //   "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  // },
  // "vocabulary": vocabulary,
  // "train_data_path": [std.format("%s_train[0:10]", dataset), std.format("%s_val[0:10]", dataset)],
  // "validation_data_path": std.format("%s_val[0:10]", dataset),
  "train_data_path": [std.format("%s", dataset), std.format("%s", dataset)],
  // "validation_data_path": std.format("%s", dataset),
  "model": {
    "type": "rsa_vqa_from_huggingface",
    "label_namespace": "answers",
    "model_name": model_name,
    "image_feature_dim": 2048,
    "image_hidden_size": 32,
    "image_num_attention_heads": 2,
    "image_num_hidden_layers": 1,
    "combined_hidden_size": 32,
    "combined_num_attention_heads": 2,
    "pooled_output_dim": pooled_output_dim,
    "image_intermediate_size": 32,
    "image_attention_dropout": 0.0,
    "image_hidden_dropout": 0.0,
    "image_biattention_id": [0],
    "text_biattention_id": [1],
    "text_fixed_layer": 0,
    "image_fixed_layer": 0,
    "fusion_method": "mul",
    "num_listener_steps": 1,
    "copy_speaker_listener": false,
    "tune_bert": false,
    "tune_images": false,
    "keep_tokens": false,
    "speaker_params": {"type": "prenorm_speaker", 
                       "encoder": {"input_size": 128,
                                   "hidden_size": pooled_output_dim,
                                   "encoder_layer": {"type": "pre_norm",
                                                     "d_model": pooled_output_dim,
                                                     "n_head": 2,
                                                     "norm": {"type": "scale_norm", 
                                                              "dim": 32},
                                                      "dim_feedforward": 64,
                                                      "dropout": 0.0,
                                                      "init_scale": 32,
                                                    }, 
                                   "num_layers": 1,
                                   "dropout": 0.0
                                   },
                       "decoder": {"type": "transformer_decoder",
                                   "input_size": 128,
                                   "hidden_size": pooled_output_dim,
                                   "decoder_layer": {"type": "pre_norm",
                                                     "d_model": pooled_output_dim,
                                                     "n_head": 2,
                                                     "norm": {"type": "scale_norm", 
                                                              "dim": 32},
                                                      "dim_feedforward": 64,
                                                      "dropout": 0.0,
                                                      "init_scale": 32,
                                                    }, 
                                    "num_layers": 1,
                                    "dropout": 0.0
                                    },
                       "dropout": 0.0},
    "listener_params": {"type": "prenorm_listener", 
                       "encoder": {"input_size": pooled_output_dim,
                                   "hidden_size": pooled_output_dim,
                                   "encoder_layer": {"type": "pre_norm",
                                                     "d_model": pooled_output_dim,
                                                     "n_head": 2,
                                                     "norm": {"type": "scale_norm", 
                                                              "dim": 32},
                                                      "dim_feedforward": 64,
                                                      "dropout": 0.0,
                                                      "init_scale": 32,
                                                    },
                                                     
                                   "num_layers": 1,
                                   "dropout": 0.0
                                   },
                        },
    
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": false,
    //[if !construct_vocab then "max_instances_in_memory"]: 10240
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    // "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "type": "warmup_gradient_descent",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
      "weight_decay": 0.01,
    },
    // "validation_metric": "+vqa_score",
    "save_warmup": 298,
    "patience": 301,
    "num_epochs": 300,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
  },
  "random_seed": 12,
  "numpy_seed": 12,
  "pytorch_seed": 12,
}