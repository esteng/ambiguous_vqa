from cProfile import label
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union
import pdb 

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
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.rsa_vqa.listener import BaseListenerModule
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.modules.vision.vision_language_encoder import VisionLanguageEncoder

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
        dropout: float = 0.1,
        vqa_loss_factor: float = 1.0,
        speaker_loss_factor: float = 1.0,
        label_namespace: str = "answers",
        keep_tokens: bool = False,
    ) -> None:
        super().__init__(vocab)

        # self.debug_tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        from allennlp.training.metrics import F1MultiLabelMeasure
        self.f1_metric = F1MultiLabelMeasure(average="micro")

        from allennlp.training.metrics.vqa import VqaMeasure
        self.vqa_metric = VqaMeasure()

        from allennlp.training.metrics import CategoricalAccuracy
        from allennlp.training.metrics import BLEU
        exclude_indices = set(speaker_module._exclude_indices) | set([0])

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

        self.classifier = torch.nn.Linear(self.vision_language_encoder.encoder.hidden_size2, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.vqa_loss_factor = vqa_loss_factor
        self.speaker_loss_factor = speaker_loss_factor

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
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        speaker_encoder_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()
        
        if speaker_encoder_outputs is None:
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

            speaker_encoder_outputs = [None for i in range(self.num_listener_steps)]
        else:
            listener_output = None
            question_input = None

        speaker_losses = []
        speaker_outputs = []   
        for i in range(self.num_listener_steps): 
            if listener_output is None:
                bsz = speaker_encoder_outputs[i].shape[0]
            else:
                bsz = listener_output.shape[0]
            speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                     gold_utterance=question_input,
                                                     speaker_encoder_output=speaker_encoder_outputs[i])
                                                    #  gold_utterance_output=question_output)
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
                true = self.speaker_modules[i].make_output_human_readable(gold_predictions)['predicted_tokens']

                # logger.info("")
                # logger.info(f"pred: {pred[0]}")
                # logger.info(f"true: {true[0]}")
                # logger.info(f"pred: {pred[1]}")
                # logger.info(f"true: {true[1]}")
            else:
                pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
                speaker_utterances.append(pred)

            encoded_by_speaker = speaker_output['encoder_output']['encoder_outputs']
            speaker_outputs.append(encoded_by_speaker)
            if question_input is None:
                if speaker_encoder_outputs[i] is None:
                    beam_size = self.speaker_modules[i].decoder._beam_search.beam_size
                else:
                    beam_size = 1 
                encoded_by_speaker = encoded_by_speaker.reshape((bsz, beam_size, -1)) 

            listener_mask = torch.ones_like(encoded_by_speaker)[:,:,0]
            listener_output = self.listener_modules[i](encoded_by_speaker,
                                                       listener_mask) 


        logits = self.classifier(listener_output['output']) 
        probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs, "speaker_loss": speaker_losses, "speaker_outputs": speaker_outputs, "speaker_utterances": speaker_utterances}
        if labels is not None and label_weights is not None:
            label_mask = labels > 1  # 0 is padding, 1 is OOV, which we want to ignore

            weighted_labels = util.masked_index_replace(
                logits.new_zeros(logits.size() + (1,)),
                labels.clamp(min=0),
                label_mask,
                label_weights.unsqueeze(-1),
            ).squeeze(-1)

            # TODO: (elias): with a different label output vocab, is this still true? 
            # weighted_labels now has shape (batch_size, num_labels).  We need to ignore the first
            # two columns of this in our loss function and accuracy metric.  The first column is a
            # padding label, and the second column is an OOV label.  We want the loss function to
            # be computed on every other label.
            binary_label_mask = weighted_labels.new_ones(logits.size())
            binary_label_mask[:, 0] = 0
            binary_label_mask[:, 1] = 0

            # pdb.set_trace() 
            vqa_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, weighted_labels, weight=binary_label_mask, reduction="sum"
                )
                / batch_size
            )
            # if vqa_loss.item() < 0.2:
            #     pdb.set_trace()


            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, labels, label_weights)
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
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]

        for i in range(len(self.acc_metrics)):
            result[f'speaker_bleu_{i}'] = self.acc_metrics[i].get_metric(reset)['BLEU'] 

        return result

    def eval_for_gen(self):
        self.eval()