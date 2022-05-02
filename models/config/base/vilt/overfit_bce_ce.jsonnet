local model_name = 'bert-base-uncased';
local other_model_name = 'bert-base-uncased';
local vocab_size = 30522;  // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local effective_batch_size = 256;
local gpu_batch_size = 256;
local num_gpus = 1;
local line_limit = 16;
// local line_limit = 16;

local construct_vocab = false;
// local dataset = "balanced_real_train";
local dataset = "unittest";
local dataset_vocab = "balanced_real";

local pooled_output_dim = 768;

local vilt_path = "/brtx/605-nvme1/estengel/annotator_uncertainty/models/finetune_vilt/best.ckpt";

local tokenizer = {
  type: 'pretrained_transformer',
  model_name: model_name,
};

local pretrained_token_indexers = {
  tokens: {
    type: 'pretrained_transformer',
    model_name: model_name,
  },
};
local token_indexers = { tokens: {
  type: 'single_id',
  namespace: 'target_tokens',
} };

local vilt_model ={
      type: 'vilt',
      model_name: '/brtx/605-nvme1/estengel/annotator_uncertainty/models/finetune_vilt_pytorch/',
    };

{
  dataset_reader: {
    type: 'vqav2',
    // image_dir: 'data/vqa/balanced_real',
    image_dir: std.format('/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s', dataset),
    // "tokenizer": tokenizer,
    source_token_indexers: pretrained_token_indexers,
    target_token_indexers: token_indexers,
    max_instances: line_limit,
    image_processing_batch_size: 16,
    run_image_feature_extraction: false,
    pass_raw_image_paths: true,
    multiple_answers_per_question: true,
    use_multilabel: true,
    is_training: true,
    is_validation: false,
    use_precompute: false,
    pretrained_model: vilt_model,
  },
  validation_dataset_reader: {
    type: 'vqav2',
    // image_dir: 'data/vqa/balanced_real',
    image_dir: std.format('/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/%s', dataset),
    tokenizer: tokenizer,
    source_token_indexers: pretrained_token_indexers,
    target_token_indexers: token_indexers,
    max_instances: line_limit,
    use_multilabel: true,
    is_training: false,
    is_validation: true,
    image_processing_batch_size: 16,
    multiple_answers_per_question: true,
    run_image_feature_extraction: false,
    pass_raw_image_paths: true,
    use_precompute: false,
    pretrained_model: vilt_model,
  },
  train_data_path: [std.format('%s', dataset)],
  validation_data_path: std.format('%s', dataset),
  //   "train_data_path": [std.format("%s_train", dataset)],
  //   "validation_data_path": std.format("%s_val[0:1000]", dataset),
  datasets_for_vocab_creation: [
    "train"
  ],
  vocabulary: {
    min_count: {
      target_tokens: 1,
      answers: 0,
    },
  },
  model: {
    type: 'rsa_vqa',
    label_namespace: 'answers',
    loss: {"type": "bce_ce"},
    vision_language_encoder: vilt_model, 
    num_listener_steps: 1,
    copy_speaker_listener: false,
    pooled_output_dim: pooled_output_dim,
    keep_tokens: false,
    vqa_loss_factor: 10,
    speaker_loss_factor: 1,
    speaker_module:
      {
        type: 'simple_speaker',
        target_namespace: 'target_tokens',
        encoder_in_dim: pooled_output_dim,
        encoder_num_layers: 2,
        encoder_hidden_dim: pooled_output_dim,
        encoder_dropout: 0.0,
        encoder_activation: 'relu',
        decoder: {
          type: 'auto_regressive_seq_decoder',
          decoder_net:
            {
              type: 'stacked_self_attention',
              decoding_dim: 768,
              target_embedding_dim: 768,
              feedforward_hidden_dim: 128,
              num_layers: 2,
              num_attention_heads: 2,
              dropout_prob: 0.0,
              residual_dropout_prob: 0.0,
              attention_dropout_prob: 0.0,
            },
          max_decoding_steps: 50,
          target_namespace: 'target_tokens',
          target_embedder: {
            vocab_namespace: 'target_tokens',
            embedding_dim: 768,
          },
          scheduled_sampling_ratio: 0.5,
          beam_size: 1,
        },
        dropout: 0.0,
      },
    listener_module: {
      type: 'simple_listener',
      encoder_in_dim: pooled_output_dim,
      encoder_num_layers: 2,
      encoder_hidden_dim: pooled_output_dim,
      encoder_dropout: 0.0,
      encoder_activation: 'relu',

    },

  },
  data_loader: {
    batch_size: gpu_batch_size,
    shuffle: false,
    //[if !construct_vocab then "max_instances_in_memory"]: 10240
  },
  [if num_gpus > 1 then 'distributed']: {
    cuda_devices: std.range(0, num_gpus - 1),
    // "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then 'trainer']: {
    type: 'warmup_gradient_descent',
    cuda_device: 0,
    optimizer: {
      type: 'huggingface_adamw',
      lr: 1e-3,
      correct_bias: true,
      weight_decay: 0.01,
      parameter_groups: [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [['^embeddings\\.', '^encoder.layers1\\.', '^t_pooler\\.'], { lr: 4e-3 }],
      ],
    },
    // "learning_rate_scheduler": {
    //   "type": "linear_with_warmup",
    //   //"num_steps_per_epoch": std.ceil(0 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
    //   "warmup_steps": 4000
    // },
    validation_metric: '+vqa_score',
    save_warmup: 900,
    patience: 1000,
    num_epochs: 1000,
    num_gradient_accumulation_steps: effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
