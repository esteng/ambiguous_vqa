
import os 
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union
import pdb 
from pathlib import Path
import torch.nn as nn
from overrides import overrides
import torch
from transformers import AutoModel, AutoTokenizer

from allennlp.data import TextFieldTensors, Vocabulary, TokenIndexer
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
)
from allennlp.nn import util
from allennlp.common.params import Params
from allennlp.modules.rsa_vqa.vqa_classifier import VQAClassifier
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.rsa_vqa.listener import BaseListenerModule
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.modules.vision.vision_language_encoder import CLIPLanguageEncoder, VisionLanguageEncoder, ViLTLanguageEncoder

from allennlp.nn.losses import Loss, BCELoss, WeightedBCELoss, MultilabelCELoss, AsymmetricLossMultiLabel, CELoss

logger = logging.getLogger(__name__)

@Model.register("rsa_vqa")
@Model.register("rsa_vqa_from_huggingface", constructor="from_huggingface_model_name")
class RSAVQAModel(Model):
    """
    The model
    """
    def __init__(
        self,
        vocab: Vocabulary,
        vision_language_encoder: VisionLanguageEncoder,
        speaker_module: BaseSpeakerModule,
        listener_module: BaseListenerModule,
        num_listener_steps: int,
        copy_speaker_listener: bool,
        pooled_output_dim: int,
        beam_size: int = 5,
        dropout: float = 0.1,
        loss: Loss = CELoss(),
        vqa_loss_factor: float = 1.0,
        speaker_loss_factor: float = 1.0,
        label_namespace: str = "answers",
        keep_tokens: bool = False,
        meaning_vector_source: str = "listener", 
    ) -> None:
        super().__init__(vocab)

        # self.debug_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        from allennlp.training.metrics import F1MultiLabelMeasure, BCEF1MultiLabelMeasure
        self.loss_fxn = loss
        # if isinstance(self.loss_fxn, BCELoss) or isinstance(self.loss_fxn, WeightedBCELoss) \
        #     or isinstance(self.loss_fxn, MultilabelCELoss): 
        #     self.f1_metric = BCEF1MultiLabelMeasure() 
        # else:
        self.f1_metric = F1MultiLabelMeasure(average="micro")

        from allennlp.training.metrics.vqa import VqaMeasure
        self.vqa_metric = VqaMeasure()

        from allennlp.training.metrics import CategoricalAccuracy
        from allennlp.training.metrics import BLEU
        exclude_indices = set(speaker_module._exclude_indices) | set([0])

        self.beam_size = beam_size
        speaker_module.decoder._beam_search.beam_size = beam_size 
        if copy_speaker_listener:
            speaker_modules = [deepcopy(speaker_module) for i in range(num_listener_steps)]
            listener_modules = [deepcopy(listener_module) for i in range(num_listener_steps)]
        else:
            speaker_modules = [speaker_module for i in range(num_listener_steps)]
            listener_modules = [listener_module for i in range(num_listener_steps)]

        self.acc_metrics = [BLEU(exclude_indices=exclude_indices) for i in range(len(speaker_modules))]

        self.vision_language_encoder = vision_language_encoder

        self.keep_tokens = keep_tokens
        if keep_tokens:
            self.encoded_token_projection = torch.nn.Linear(self.vision_language_encoder.encoder.hidden_size1, pooled_output_dim)

        self.speaker_modules = torch.nn.ModuleList(speaker_modules)
        self.listener_modules = torch.nn.ModuleList(listener_modules)
        self.copy_speaker_listener = copy_speaker_listener 
        self.num_listener_steps = len(speaker_modules)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.num_labels = num_labels
        self.label_namespace = label_namespace

        if isinstance(vision_language_encoder, CLIPLanguageEncoder) or isinstance(self.vision_language_encoder, ViLTLanguageEncoder): 
            self.classifier = VQAClassifier(self.vision_language_encoder.projection_dim, num_labels)
        else:
            self.classifier = VQAClassifier(self.vision_language_encoder.encoder.hidden_size2, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.vqa_loss_factor = vqa_loss_factor
        self.speaker_loss_factor = speaker_loss_factor
        
        self.meaning_vector_source = meaning_vector_source

    def _cache_meaning_vectors(self, meaning_vectors, precompute_metadata):
        checkpoint_dir = os.environ['CHECKPOINT_DIR']


        for layer in range(len(meaning_vectors)):
            for i in range(len(meaning_vectors[layer])):
                metadata = precompute_metadata[i]
                out_dir = Path(metadata['save_dir'])
                out_dir.mkdir(exist_ok=True, parents=True)
                checkpoint_file = out_dir.joinpath("checkpoint_info.txt")
                # save checkpoint info to make organization easier later 
                if not checkpoint_file.exists():
                    with open(checkpoint_file, 'w') as f:
                        f.write(str(checkpoint_dir))
                filename = out_dir.joinpath(f"{metadata['image_id']}_{metadata['question_id']}_{layer}.pt")
                if filename.exists():
                    continue
                else:
                    torch.save(meaning_vectors[layer][i], filename)

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        question: TextFieldTensors,
        question_input: torch.Tensor = None,
        # question_output: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
        labels_for_metric: Optional[torch.Tensor] = None,
        weights_for_metric: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        meaning_vectors_input: Optional[List[torch.Tensor]] = None,
        pooled_output: Optional[torch.Tensor] = None,
        sequence_output: Optional[torch.Tensor] = None,
        precompute_metadata: Optional[MetadataField] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()

        # only run vision-language encoder if the reps haven't been pre-computed  
        if meaning_vectors_input is None and pooled_output is None:
            if isinstance(self.vision_language_encoder, CLIPLanguageEncoder) or isinstance(self.vision_language_encoder, ViLTLanguageEncoder):
                # TODO (elias) remove after debugging 
                with torch.no_grad() :
                    pooled_output, sequence_output = self.vision_language_encoder(debug_tokens,
                                                                                debug_images)

            else: 
                pooled_output, sequence_output_t = self.vision_language_encoder(
                                                        box_features=box_features,
                                                        box_coordinates=box_coordinates,
                                                        question=question,
                                                        question_input=question_input)
            if self.keep_tokens:
                encoded_tokens = self.encoded_token_projection(sequence_output_t)
                listener_output = torch.cat([pooled_output.unsqueeze(1), encoded_tokens], dim=1)
            else:
                listener_output = pooled_output

        elif pooled_output is not None:
            listener_output = pooled_output
        else:
            listener_output = None
            question_input = None

        if meaning_vectors_input is None:
            meaning_vectors_input = [None for i in range(self.num_listener_steps)]
        else:
            listener_output = None
            question_input = None

        speaker_losses = []
        meaning_vectors_output = []   
        for i in range(self.num_listener_steps): 
            if listener_output is None:
                bsz = meaning_vectors_input[i].shape[0]
            else:
                bsz = listener_output.shape[0]
            
            if self.meaning_vector_source == "listener" and meaning_vectors_input[i] is None: 
                # meaning vector is set to listener output 
                meaning_vectors_output.append(listener_output)
                # run speaker module as normal, no modified output 
                speaker_output = self.speaker_modules[i](fused_representation=listener_output, 
                                                        gold_utterance=question_input)

            elif self.meaning_vector_source == "listener" and meaning_vectors_input[i] is not None:
                # run speaker module with the modified listener output 
                listener_output = meaning_vectors_input[i].squeeze(0)
                # listener output is the meaning vector 
                meaning_vectors_output.append(listener_output)
                speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                        gold_utterance=question_input,
                                                        do_detach=True)
            else:
                # using speaker encoder output as meaning vector
                speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                        gold_utterance=question_input,
                                                        speaker_encoder_output=meaning_vectors_input[i])


            speaker_loss = speaker_output['loss']
            speaker_losses.append(speaker_loss) 
            speaker_utterances = []
            if question_input is not None: 
                speaker_predictions = speaker_output['predictions'].contiguous() 
                # get index of EOS 
                prediction_mask = torch.zeros_like(speaker_predictions)
                eos_prediction_indices = speaker_predictions == self.speaker_modules[i]._end_index 
                prediction_mask[eos_prediction_indices] = 1 
                prediction_mask = torch.cumsum(prediction_mask, dim=1).clamp(min=0, max=1)
                # keep EOS in
                prediction_mask[eos_prediction_indices] = 0
                prediction_mask = 1 - prediction_mask

                # mask out everything that comes after EOS 
                speaker_predictions *= prediction_mask 
                self.acc_metrics[i](speaker_predictions,
                                    question_input['tokens']['tokens'][:,1:].contiguous())

                gold_predictions = {"predictions": question_input['tokens']['tokens'][:,1:]}
                pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
                speaker_utterances.append(pred)
                true = self.speaker_modules[i].make_output_human_readable(gold_predictions)['predicted_tokens']

            else:
                pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
                speaker_utterances.append(pred)

            encoded_by_speaker = speaker_output['encoder_output']['encoder_outputs']

            if self.meaning_vector_source == "speaker": 
                meaning_vectors_output.append(encoded_by_speaker)

            if question_input is None:
                if meaning_vectors_input[i] is None: 
                    beam_size = self.speaker_modules[i].decoder._beam_search.beam_size
                else:
                    beam_size = 1 
                encoded_by_speaker = encoded_by_speaker.reshape((bsz, beam_size, -1)) 

            listener_mask = torch.ones_like(encoded_by_speaker)[:,:,0]
            listener_output = self.listener_modules[i](encoded_by_speaker,
                                                       listener_mask) 

        logits = self.classifier(listener_output['output']) 
        probs = torch.softmax(logits, dim=-1)

        predicted_labels = torch.argmax(logits, dim=1)

        outputs = {"logits": logits, "probs": probs, "speaker_loss": speaker_losses, 
                   "meaning_vectors_output": meaning_vectors_output, "speaker_utterances": speaker_utterances,
                   "predicted_labels": predicted_labels}

        if labels is not None and label_weights is not None:
            vqa_loss, weighted_labels, mask = self.loss_fxn(logits, labels,  debug_answer, label_weights=label_weights)
            outputs['debug_answer'] = debug_answer
            if isinstance(self.loss_fxn, BCELoss) or isinstance(self.loss_fxn, WeightedBCELoss): 

                binary_label_mask = mask 
                self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
                self.vqa_metric(logits, labels_for_metric, weights_for_metric, do_interact=False, debug_tokens=debug_tokens, debug_answer=debug_answer) 

            elif isinstance(self.loss_fxn, MultilabelCELoss):  
                labels = labels.squeeze(1)
                labels_for_metric = torch.argmax(labels, dim=1)
                label_weights_for_f1_metric = torch.ones_like(labels_for_metric)
            else:
                labels_for_metric = labels 
                label_weights_for_metric = label_weights

            # logger.info(f"loss: {loss.item()}")
            # speaker_loss_sum = torch.sum(torch.Tensor(speaker_losses))
            # logger.info(f"speaker loss: {speaker_loss_sum.item()}")
            losses = [self.vqa_loss_factor * vqa_loss] + [self.speaker_loss_factor * x for x in speaker_losses]
            # losses = speaker_losses
            big_loss = 0.0
            for loss in losses:
                big_loss += loss

            outputs['loss'] =  big_loss
            outputs['vqa_loss'] = vqa_loss

            # self.vqa_metric(logits, labels_for_metric, label_weights_for_metric, do_interact = vqa_loss.item() < 50)
            # if not isinstance(self.loss_fxn, CELoss):
            #     # self.f1_metric(logits, labels)
            #     self.f1_metric(logits, labels_for_f1_metric, weight_for_metric.bool())

        return outputs

    def softmax_cross_entropy_with_softtarget(self, input, target, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]

        for i in range(len(self.acc_metrics)):
            result[f'speaker_bleu_{i}'] = self.acc_metrics[i].get_metric(reset)['BLEU'] 

        return result

    def eval_for_gen(self):
        self.eval()


@Model.register("precompute_vqa")
@Model.register("precompute_from_huggingface", constructor="from_huggingface_model_name")
class PrecomputeVQAModel(RSAVQAModel):
    """
    Keep everything the same so that we can use the same config, but most things here don't get used 
    """
    def __init__(
        self,
        vocab: Vocabulary,
        vision_language_encoder: VisionLanguageEncoder,
        speaker_module: BaseSpeakerModule,
        listener_module: BaseListenerModule,
        num_listener_steps: int,
        copy_speaker_listener: bool,
        pooled_output_dim: int,
        dropout: float = 0.1,
        losses: str = "ce",
        vqa_loss_factor: float = 1.0,
        speaker_loss_factor: float = 1.0,
        label_namespace: str = "answers",
        keep_tokens: bool = False,
    ) -> None:
        super().__init__(vocab,
                         vision_language_encoder,
                         speaker_module,
                         listener_module,
                         num_listener_steps,
                         copy_speaker_listener,
                         pooled_output_dim,
                         dropout,
                         losses,
                         vqa_loss_factor,
                         speaker_loss_factor,
                         label_namespace,
                         keep_tokens)


    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        question: TextFieldTensors,
        question_input: torch.Tensor = None,
        # question_output: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
        label_indices: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        precompute_metadata: Optional[MetadataField] = None,
        speaker_encoder_outputs: Optional[List[torch.Tensor]] = None,
        pooled_output: Optional[torch.Tensor] = None,
        sequence_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()

        if isinstance(self.vision_language_encoder, CLIPLanguageEncoder) or isinstance(self.vision_language_encoder, ViLTLanguageEncoder):
            # TODO (elias) remove after debugging 
            with torch.no_grad() :
                pooled_output, sequence_output = self.vision_language_encoder(debug_tokens,
                                                                            debug_images)

        else: 
            pooled_output, sequence_output_t = self.vision_language_encoder(
                                                    box_features=box_features,
                                                    box_coordinates=box_coordinates,
                                                    question=question,
                                                    question_input=question_input)


        for i in range(pooled_output.shape[0]):
            metadata = precompute_metadata[i]
            out_dir = Path(metadata['save_dir'])
            out_dir.mkdir(exist_ok=True, parents=True)
            filename = out_dir.joinpath(f"{metadata['image_id']}_{metadata['question_id']}.pt")
            if filename.exists():
                continue
            else:
                torch.save(pooled_output[i], filename)

        return {"loss": torch.zeros(1, requires_grad=True)}
