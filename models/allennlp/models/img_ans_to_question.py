
import os 
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union
import pdb 
from pathlib import Path
from overrides import overrides
import torch
import numpy as np 

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model

from allennlp.modules.rsa_vqa.vqa_classifier import VQAClassifier
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.vision.vision_language_encoder import CLIPLanguageEncoder, VisionLanguageEncoder, ViLTLanguageEncoder

from allennlp.nn.losses import (Loss, 
                                BCELoss, 
                                WeightedBCELoss, 
                                MultilabelCELoss, 
                                CELoss, 
                                CEAndBCELoss)

logger = logging.getLogger(__name__)

@Model.register("img_ans_2_question")
@Model.register("img_ans_2_question_from_huggingface", constructor="from_huggingface_model_name")
class ImageAnswer2QuestionModel(Model):
    """
    The model to go from Image + Answer to Question 
    """
    def __init__(
        self,
        vocab: Vocabulary,
        vision_language_encoder: VisionLanguageEncoder,
        question_output_module: BaseSpeakerModule,
        pooled_output_dim: int,
        beam_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)

        from allennlp.training.metrics import BLEU
        exclude_indices = set(question_output_module._exclude_indices) | set([0])
        self.acc_metrics = BLEU(exclude_indices=exclude_indices) 

        self.vision_language_encoder = vision_language_encoder
        self.question_output_module = question_output_module

        self.pooled_output_projection = torch.nn.Linear(pooled_output_dim, question_output_module.encoder_in_dim)

        self.num_labels = len(self.vision_language_encoder.model.config.label2id)

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def forward(
        self,  # type: ignore
        question: TextFieldTensors,
        answers_as_text: MetadataField,
        answer_counts: torch.Tensor,
        question_id: MetadataField,
        question_output: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            pooled_output, __ = \
                self.vision_language_encoder(answers_as_text,
                                            debug_images)

        pooled_output = self.pooled_output_projection(pooled_output)

        generated_output = self.question_output_module(fused_representation = pooled_output,
                                                      gold_utterance = question_output)
        speaker_utterances = []
        question_predictions = generated_output['predictions'].contiguous() 
        self.acc_metrics(question_predictions,
                         question_output['tokens']['tokens'][:,1:].contiguous())
        if not self.training:
            # pdb.set_trace()
            pred = self.question_output_module.make_output_human_readable(generated_output)['predicted_tokens']
            speaker_utterances.append(pred)

        loss = generated_output['loss']
        outputs = {"question_id": question_id, 
                   "loss": loss, 
                   "speaker_utterances": speaker_utterances}
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.acc_metrics.get_metric(reset)
        return result

    def eval_for_gen(self):
        self.eval()

