
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
from allennlp.modules.vision.vision_language_encoder import (
    CLIPLanguageEncoder, 
    VisionLanguageEncoder, 
    ViLTLanguageEncoder,
    ClassifierViLTLanguageEncoder
)

from allennlp.nn.losses import (
    Loss, 
    BCELoss, 
    WeightedBCELoss, 
    MultilabelCELoss, 
    AsymmetricLossMultiLabel, 
    CELoss
) 

logger = logging.getLogger(__name__)

@Model.register("rsa_debug")
@Model.register("rsa_debug_from_huggingface", constructor="from_huggingface_model_name")
class DebugRSAVQAModel(Model):
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
        # No speaker or listener to debug 
        # if copy_speakekr_listener:
        #     speaker_modules = [deepcopy(speaker_module) for i in range(num_listener_steps)]
        #     listener_modules = [deepcopy(listener_module) for i in range(num_listener_steps)]
        # else:
        #     speaker_modules = [speaker_module for i in range(num_listener_steps)]
        #     listener_modules = [listener_module for i in range(num_listener_steps)]

        self.vision_language_encoder = vision_language_encoder

        self.keep_tokens = keep_tokens
        if keep_tokens:
            self.encoded_token_projection = torch.nn.Linear(self.vision_language_encoder.encoder.hidden_size1, pooled_output_dim)

        # self.speaker_modules = torch.nn.ModuleList(speaker_modules)
        # self.listener_modules = torch.nn.ModuleList(listener_modules)
        # self.copy_speaker_listener = copy_speaker_listener 
        # self.num_listener_steps = len(speaker_modules)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.num_labels = num_labels
        self.label_namespace = label_namespace

        if isinstance(vision_language_encoder, CLIPLanguageEncoder) \
            or isinstance(self.vision_language_encoder, ViLTLanguageEncoder)\
            or isinstance(self.vision_language_encoder, ClassifierViLTLanguageEncoder): 
            self.classifier = VQAClassifier(self.vision_language_encoder.projection_dim, num_labels)
        else:
            self.classifier = VQAClassifier(self.vision_language_encoder.encoder.hidden_size2, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.vqa_loss_factor = vqa_loss_factor
        
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

    def convert_labels(self, labels): 
        new_labels = torch.ones_like(labels) * -1
        for i in range(labels.size(0)):
            for j in range(labels.size(1)):
                try:
                    tok = self.vocab.get_token_from_index(labels[i, j].item(), namespace=self.label_namespace)
                    vilt_label = self.vision_language_encoder.model.config.label2id[tok]
                except KeyError:
                    vilt_label = -1
                new_labels[i, j] = vilt_label
        return torch.tensor(new_labels) 

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
        labels_from_pretrained: Optional[torch.Tensor] = None,
        weights_for_metric: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        meaning_vectors_input: Optional[List[torch.Tensor]] = None,
        pooled_output: Optional[torch.Tensor] = None,
        sequence_output: Optional[torch.Tensor] = None,
        precompute_metadata: Optional[MetadataField] = None,
    ) -> Dict[str, torch.Tensor]:


        # with torch.no_grad():
        self.vision_language_encoder.eval()
        logits = self.vision_language_encoder(debug_tokens,
                                            debug_images)


        probs = torch.softmax(logits, dim=-1)

        predicted_labels = torch.argmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs, "speaker_loss": 0.0, 
                   "meaning_vectors_output": [], "speaker_utterances": [],
                   "predicted_labels": predicted_labels}



        if labels_from_pretrained is not None and label_weights is not None:
            # labels = self.convert_labels(labels)
            vqa_loss, weighted_labels, mask = self.loss_fxn(logits, labels_from_pretrained,  debug_answer, label_weights=label_weights)
            outputs['debug_answer'] = debug_answer

            binary_label_mask = mask 
            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, labels_from_pretrained, weights_for_metric, do_interact=False, debug_tokens=debug_tokens, debug_answer=debug_answer) 

            outputs['loss'] =  vqa_loss 
            outputs['vqa_loss'] = vqa_loss 

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

        return result

    def eval_for_gen(self):
        self.eval()

