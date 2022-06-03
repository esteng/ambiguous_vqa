# """
# The `evaluate` subcommand can be used to
# evaluate a trained model against a dataset
# and report any metrics calculated by the model.
# """

# import pdb 
# import argparse
# import json
# import logging
# from typing import Any, Dict

# from overrides import overrides
# import torch 

# from allennlp.commands.subcommand import Subcommand
# from allennlp.common import logging as common_logging
# from allennlp.common.util import prepare_environment
# from allennlp.data import DataLoader
# from allennlp.models.archival import load_archive
# from allennlp.training.util import evaluate, minimize_and_generate, baseline_save_and_generate

# logger = logging.getLogger(__name__)


# @Subcommand.register("min_gen")
# class MinGen(Subcommand):
#     @overrides
#     def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
#         description = """Evaluate the specified model + dataset"""
#         subparser = parser.add_parser(
#             self.name, description=description, help="Evaluate the specified model + dataset."
#         )

#         subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

#         subparser.add_argument(
#             "input_file", type=str, help="path to the file containing the evaluation data"
#         )

#         subparser.add_argument(
#             "--lr", type=float, default=0.01, help="learn rate for optimizing the encoded vector")
#         subparser.add_argument(
#             "--descent-strategy", type=str, choices=['steps', 'thresh'], default='steps', help = "stopping condition for the descent")
#         subparser.add_argument(
#             "--descent-loss-threshold", type=float, default=0.05, help = "number of steps to optimize")
#         subparser.add_argument(
#             "--num-descent-steps", type=int, default=10, help = "number of steps to optimize")
#         subparser.add_argument(
#             "--output-file", type=str, help="optional path to write the metrics to as JSON"
#         )
#         subparser.add_argument(
#             "--mix-strategy", type=str, choices=[None, "end", "continuous"], default=None, help = "mix strategy the vectors. None means no mixing, end means mixing only at the end, continous mixes in the original meaning vector at each optimization step "
#         )
#         subparser.add_argument(
#             "--mix-ratio", type=float, default=0.5, help="mix ratio"
#         )

#         subparser.add_argument(
#             "--predictions-output-file",
#             type=str,
#             help="optional path to write the predictions to as JSON lines",
#         )

#         subparser.add_argument(
#             "--weights-file", type=str, help="a path that overrides which weights file to use"
#         )

#         cuda_device = subparser.add_mutually_exclusive_group(required=False)
#         cuda_device.add_argument(
#             "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
#         )

#         subparser.add_argument(
#             "-o",
#             "--overrides",
#             type=str,
#             default="",
#             help=(
#                 "a json(net) structure used to override the experiment configuration, e.g., "
#                 "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
#                 " with nested dictionaries or with dot syntax."
#             ),
#         )
#         subparser.add_argument(
#             "--beam-size", type=int, help="Beam size when using beam search decoder", default=5,
#         )

#         subparser.add_argument(
#             "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
#         )

#         subparser.add_argument(
#             "--batch-weight-key",
#             type=str,
#             default="",
#             help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
#         )

#         subparser.add_argument(
#             "--extend-vocab",
#             action="store_true",
#             default=False,
#             help="if specified, we will use the instances in your new dataset to "
#             "extend your vocabulary. If pretrained-file was used to initialize "
#             "embedding layers, you may also need to pass --embedding-sources-mapping.",
#         )

#         subparser.add_argument(
#             "--embedding-sources-mapping",
#             type=str,
#             default="",
#             help="a JSON dict defining mapping from embedding module path to embedding "
#             "pretrained-file used during training. If not passed, and embedding needs to be "
#             "extended, we will try to use the original file paths used during training. If "
#             "they are not available we will use random vectors for embedding extension.",
#         )
#         subparser.add_argument(
#             "--file-friendly-logging",
#             action="store_true",
#             default=False,
#             help="outputs tqdm status on separate lines and slows tqdm refresh rate",
#         )
#         subparser.add_argument(
#             "--precompute-intermediate",
#             action="store_true",
#             default=False,
#             help="precompute intermediate meaning vectors and store in a file "
#         )
#         subparser.add_argument(
#             "--retrieval-save-dir",
#             type=str,
#             default=None,
#             help="path to store precomputed train representations for retrieval"
#         ) 
#         subparser.add_argument(
#             "--beta-text-loss",
#             type=float,
#             default=0.0,
#             help="hyperparam for how much to weight inverse text generation loss"
#         )

#         subparser.set_defaults(func=min_gen_from_args)

#         return subparser


# def min_gen_from_args(args: argparse.Namespace) -> Dict[str, Any]:
#     common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

#     # Disable some of the more verbose logging statements
#     logging.getLogger("allennlp.common.params").disabled = True
#     logging.getLogger("allennlp.nn.initializers").disabled = True
#     logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

#     # Load from archive
#     archive = load_archive(
#         args.archive_file,
#         weights_file=args.weights_file,
#         cuda_device=args.cuda_device,
#         overrides=args.overrides,
#     )
#     config = archive.config
#     prepare_environment(config)
#     model = archive.model
#     model.eval_for_gen()

#     # Load the evaluation data

#     dataset_reader = archive.validation_dataset_reader
#     if args.precompute_intermediate:
#         dataset_reader.retrieval_baseline = True
#         dataset_reader.retrieval_save_dir = args.retrieval_save_dir

#     evaluation_data_path = args.input_file
#     logger.info("Reading evaluation data from %s", evaluation_data_path)

#     data_loader_params = config.pop("validation_data_loader", None)
#     if data_loader_params is None:
#         data_loader_params = config.pop("data_loader")
#     if args.batch_size:
#         data_loader_params["batch_size"] = args.batch_size
#     data_loader = DataLoader.from_params(
#         params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
#     )

#     embedding_sources = (
#         json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}
#     )

#     if args.extend_vocab:
#         logger.info("Vocabulary is being extended with test instances.")
#         model.vocab.extend_from_instances(instances=data_loader.iter_instances())
#         model.extend_embedder_vocab(embedding_sources)

#     data_loader.index_with(model.vocab)


#     if args.lr == -1:
#         metrics = baseline_save_and_generate(
#                     model,
#                     data_loader,
#                     beam_size=args.beam_size,
#                     cuda_device=args.cuda_device,
#                     output_file=args.output_file,
#                     predictions_output_file=args.predictions_output_file,
#                     precompute_intermediate=args.precompute_intermediate,
#                     retrieval_save_dir=args.retrieval_save_dir,
#         )
#     else:
#         metrics = minimize_and_generate(
#             model,
#             data_loader,
#             beam_size=args.beam_size,
#             descent_strategy=args.descent_strategy,
#             descent_loss_threshold=args.descent_loss_threshold,
#             num_descent_steps=args.num_descent_steps, 
#             mix_strategy=args.mix_strategy,
#             mix_ratio=args.mix_ratio,
#             lr = args.lr,
#             cuda_device=args.cuda_device,
#             batch_weight_key=args.batch_weight_key,
#             output_file=args.output_file,
#             predictions_output_file=args.predictions_output_file,
#             precompute_intermediate=args.precompute_intermediate,
#             retrieval_save_dir=args.retrieval_save_dir,
#             beta_text_loss=args.beta_text_loss,
#         )

#     logger.info("Finished evaluating.")

#     return metrics
